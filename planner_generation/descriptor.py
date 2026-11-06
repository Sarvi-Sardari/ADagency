"""Generate per-image descriptions and consolidated normal-class summaries."""

import re
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import DatasetConfig, ModelsConfig
from .models import llm, vlm
from .utils import build_prompt, build_prompt_blocks, preprocess_img
from pipeline_core.model_output import normalize_model_output

logger = logging.getLogger(__name__)

OBJECT_SUMMARY_MAX_ATTEMPTS = 3


def _strip_model_wrappers(text: str) -> str:
    """Remove structural control tokens emitted around otherwise valid text."""
    return normalize_model_output(text)


def _parse_json_mapping(raw: str, context: str) -> dict:
    cleaned = _strip_model_wrappers(raw)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if not cleaned.startswith("{"):
        cleaned = "{" + cleaned + "}"
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as strict_exc:
        # LLMs occasionally put a literal newline or tab inside a quoted JSON
        # value.  Python's relaxed decoder can recover that content safely;
        # the master file is subsequently re-serialized as strict JSON.
        try:
            parsed = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            raise ValueError(f"{context} returned invalid JSON: {strict_exc}") from strict_exc
        logger.warning("%s contained unescaped control characters; normalized them.", context)
    if not isinstance(parsed, dict):
        raise TypeError(f"{context} must return a JSON object, got {type(parsed).__name__}.")
    return parsed


def _parse_object_summary(raw: str, expected_key: str) -> dict:
    """Parse and validate one summarizer response before it reaches disk."""
    context = f"Object summarizer for {expected_key!r}"
    data = _parse_json_mapping(raw, context)
    if len(data) != 1:
        raise ValueError(
            f"{context} must return exactly one top-level object, got keys {list(data)}"
        )
    returned_key = _resolve_object_mapping_key(data, expected_key)
    summary = data[returned_key]
    if not isinstance(summary, dict):
        raise TypeError(
            f"{context} value must be a JSON object, got {type(summary).__name__}."
        )
    if not summary:
        raise ValueError(f"{context} returned an empty object description.")
    return summary


def _resolve_object_mapping_key(data: dict, expected_key: str) -> str:
    """Resolve harmless formatting variants while rejecting semantic renames."""
    if expected_key in data:
        return expected_key

    def normalized(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).casefold())

    expected_normalized = normalized(expected_key)
    matches = [key for key in data if normalized(key) == expected_normalized]
    match_reason = "separator/case formatting"
    if not matches:
        # Some models keep the full qualifier in the value but omit a trailing
        # parenthetical qualifier from the JSON key, e.g.
        # ``Cable Insulation Layer (White/Off-white)`` ->
        # ``Cable_Insulation_Layer``. Accept only that mechanically derivable
        # base label; do not permit arbitrary abbreviated or renamed keys.
        expected_without_parentheticals = re.sub(r"\s*\([^()]*\)\s*", " ", expected_key).strip()
        base_normalized = normalized(expected_without_parentheticals)
        if base_normalized and base_normalized != expected_normalized:
            matches = [key for key in data if normalized(key) == base_normalized]
            match_reason = "omitted parenthetical qualifier"
    if len(matches) == 1:
        logger.warning(
            "Object summarizer formatted key %r as %r (%s); preserving the configured object label.",
            expected_key,
            matches[0],
            match_reason,
        )
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"Object summarizer returned multiple formatting-equivalent keys for "
            f"{expected_key!r}: {matches}"
        )
    raise KeyError(
        f"Object summarizer JSON does not contain the expected key "
        f"{expected_key!r}: {list(data)}"
    )


def _extract_object_description(text: str, object_name: str) -> str | None:
    """Extract one object block from documented and common VLM header styles."""
    # GLM-4.6V may wrap grounded output in model control tokens even when the
    # prompt requests plain text.  Remove only those structural wrappers so
    # the strict object-label check below still rejects renamed objects.
    text = _strip_model_wrappers(text)
    escaped_name = re.escape(object_name)
    header_pattern = re.compile(
        rf"^\s*(?:[-*]\s*)?object\s*\d+\s*(?::|-)?\s*"
        rf"<?\s*{escaped_name}\s*>?\s*$"
        rf"|^\s*-\s*<?\s*{escaped_name}\s*>?\s*[.:]?\s*$"
        rf"|^\s*<?\s*{escaped_name}\s*>?\s*[.:]?\s*$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    header = header_pattern.search(text)
    if header is None:
        return None

    # Attribute lines belong to the current object.  Stop only at the next
    # numbered object heading, in either ``Object 2: name`` or prompt format.
    next_header = re.compile(
        r"^\s*(?:[-*]\s*)?object\s*\d+\b", flags=re.IGNORECASE | re.MULTILINE
    ).search(text, header.end())
    end = next_header.start() if next_header else len(text)
    return text[header.end():end].strip()


def normal_description_generator(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    vlmModel: vlm,
    list_of_ref_img_filepaths: list[str],
    dataset_name: str,
    is_class: bool,
    object_name_list: list[str],
    class_label: str | None = None,
) -> tuple[list[Path], list[Path]]:

    """desription: This function takes the reference image like good or anomaloús free image and object names of that image as list as input to give the description of the image as output and save it in a file.\n
    Input: cfg: dataclass,list_of_ref_img_filepaths:list of filepath of the good or anomolous free image,dataset_name: name of the dataset,is_class:boolean value set to true if its a multi class dataset and false if its not,object_name_list:list of the name of the objects present in the given images.\n
    Output: paths of the description JSON files and preprocessed images created
    during this invocation. Returning the paths lets downstream stages consume
    only current-run artifacts instead of scanning persistent output folders.
    """
    
    pre_processed_image_folder = datacfg.output.preprocessed_reference_dir / dataset_name
    os.makedirs(pre_processed_image_folder,exist_ok=True)
    os.makedirs(datacfg.output.descriptions_dir,exist_ok=True)
    dataset_description_folder = os.path.join(datacfg.output.descriptions_dir,dataset_name)
    os.makedirs(dataset_description_folder,exist_ok=True)
    if is_class:
        if not class_label:
            raise ValueError("class_label is required for multi-class description generation.")
        class_label_description_folder = os.path.join(dataset_description_folder, class_label)
        os.makedirs(class_label_description_folder,exist_ok=True)
        class_label_preprocessed_image_folder = os.path.join(pre_processed_image_folder, class_label)
        os.makedirs(class_label_preprocessed_image_folder,exist_ok=True)
    else:
        # The anomaly-detection pipeline resolves every single-class planner
        # under the label "good", so keep the same class-shaped layout here.
        class_label = "good"
        class_label_description_folder = os.path.join(dataset_description_folder, "good")
        os.makedirs(class_label_description_folder,exist_ok=True)
        class_label_preprocessed_image_folder = os.path.join(pre_processed_image_folder, "good")
        os.makedirs(class_label_preprocessed_image_folder,exist_ok=True)
    
    logger.debug("Preprocessed image folder: %s", class_label_preprocessed_image_folder)
    preprocessed_image_paths: list[Path] = []
    for ref_img_path in list_of_ref_img_filepaths:
        tag_name = f"{class_label}_{Path(ref_img_path).name}".split('.')[0]
        preprocessed_image_paths.append(
            preprocess_img(modelcfg=modelcfg,
                           src=ref_img_path,
                           out_dir=Path(class_label_preprocessed_image_folder),
                           tag=tag_name)
        )
    description_paths: list[Path] = []
    for full_img_path in preprocessed_image_paths:
        if full_img_path.is_file():
            img_path = full_img_path.name
            img = Image.open(full_img_path).convert("RGB")
            # Describe one configured label per call.  Small VLMs often stop
            # after the first example block or collapse overlapping part names
            # when all labels are requested together.  Isolation guarantees
            # that every label gets its own generation without dataset-specific
            # aliases or hardcoded object names.
            generated_descriptions: list[str] = []
            vlm_time = 0.0
            for object_name in object_name_list:
                prompt = build_prompt(
                    datacfg.prompts_dir,
                    "normal_descriptor.j2",
                    objects_str=object_name,
                    object_name_list=[object_name],
                )
                out, object_vlm_time = vlmModel.run_vlm(modelcfg.vlm, [img], prompt)
                vlm_time += object_vlm_time
                description = out[1]['content'].strip()
                description_block = _extract_object_description(description, object_name)
                if description_block is None:
                    raise ValueError(
                        f"Normal descriptor did not return the exact requested label "
                        f"{object_name!r} for image {full_img_path}. Raw response: {description!r}"
                    )
                # Normalize accepted header variants (for example, a bare
                # ``Leather Surface.`` line) so combined per-label responses
                # always have unambiguous boundaries for the master parser.
                generated_descriptions.append(
                    f"Object 1: {object_name}\n{description_block}".strip()
                )

            combined_description = "\n\n".join(generated_descriptions)
            filename = f'{img_path.split(".")[0]}_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.json'
            description_filepath = Path(class_label_description_folder) / filename
            with description_filepath.open('w', encoding='utf-8') as f:
                json.dump(combined_description, f, indent=4, ensure_ascii=False)
            description_paths.append(description_filepath)

            logger.info(
                "Image description saved to %s (labels=%d, inference_time=%.3fs)",
                description_filepath,
                len(object_name_list),
                vlm_time,
            )
    return description_paths, preprocessed_image_paths

def object_summariser(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    llmModel: llm,
    description_text: str,
    object_name: str,
    number_of_classes: int,
    previous_response: str | None = None,
    validation_error: str | None = None,
) -> str:

    """description: This function takes the list of object description and return the summarised version of it.\n
  Input: cfg: dataclass,description_text:list of object descriptions ,object_name:name of the object,nummber_of_classes:the number of class labels"""
    
    parts = build_prompt_blocks(
            datacfg.prompts_dir,
            "object_summarizer.j2",
            object_name=object_name,
            description_text=description_text,
            previous_response=previous_response,
            validation_error=validation_error,
        )
    system_message = parts["system"]
    prompt = parts["user"]
    out, llm_time = llmModel.run_llm(modelcfg.llm, system_message,prompt)
    logger.info("[%s] Object summarisation completed in %.3fs", object_name, llm_time)
    pattern = r'<think>.*?</think>'
    final_text = re.sub(pattern, '', out[-1]['content'], flags=re.DOTALL)
    return final_text

def object_description_processing(img_description_list:list[dict[str,str]])->dict[str,list[str]]:

    """description: this fucntion takes the all the object description of the objects separeately and groups them in a dictionary where key is the object name and value is the list of all object descriptions.\n
    Input: img_description_list: list of object description \n
    Output: dictionary with key as object name and value as list of all the object descriptions.\n"""

    list_objects = []
    for image_description in img_description_list:
        list_objects.extend(image_description.keys())
    list_objects = set(list_objects)
    logger.debug("Unique object keys found: %s", list_objects)

    result = {}
    for key in list_objects:
        list_value = []
        for image_description in img_description_list:
            if key in image_description.keys():
                list_value.append(image_description[key])
        result[key] = list_value
    # print(result)
    return result

def master_description_file_writer(datacfg: DatasetConfig, modelcfg: ModelsConfig, llmModel: llm, object_description:dict[str,list[str]], class_directory_path:Path, class_label:str)->Path:

    """description: This fucntion takes the dictionary that has the object name as the key and value the list of all the oject description passes the list of description to a summariser fucntion to summarise it and saves the summary of such object and combines each object description into master description file\n
    Input: object_description: dictionary with key as object name and value as list of its description, class_directory_path: directory path in which the master description will be saved., class label: class label of the dataset.\n
    """
    
    master_description = {}
    for key, value in object_description.items():
        description_text = f'{key}:{value}'
        previous_response = None
        validation_error = None
        summary = None
        for attempt in range(1, OBJECT_SUMMARY_MAX_ATTEMPTS + 1):
            result = object_summariser(
                datacfg=datacfg,
                modelcfg=modelcfg,
                llmModel=llmModel,
                description_text=description_text,
                object_name=key,
                number_of_classes=3,
                previous_response=previous_response,
                validation_error=validation_error,
            )
            logger.debug(
                "Summarised description for '%s' (attempt %d/%d): %s",
                key,
                attempt,
                OBJECT_SUMMARY_MAX_ATTEMPTS,
                result,
            )
            try:
                summary = _parse_object_summary(result, key)
                break
            except (KeyError, TypeError, ValueError) as exc:
                if attempt == OBJECT_SUMMARY_MAX_ATTEMPTS:
                    raise ValueError(
                        f"Object summarizer for {key!r} failed validation after "
                        f"{OBJECT_SUMMARY_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc
                validation_error = str(exc)
                previous_response = result
                logger.warning(
                    "Object summarizer for %r failed validation on attempt %d/%d: %s; "
                    "retrying with corrective feedback.",
                    key,
                    attempt,
                    OBJECT_SUMMARY_MAX_ATTEMPTS,
                    exc,
                )
        assert summary is not None
        master_description[key] = summary
    logger.debug("Master description: %s", master_description)
    master_description_filepath = os.path.join(class_directory_path,f'{class_label}_{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}_master_description.json')
    with open(master_description_filepath, 'w', encoding='utf-8') as f:
        json.dump(master_description, f, indent=4, ensure_ascii=False)
    return Path(master_description_filepath)

def master_description_generator(
    datacfg: DatasetConfig,
    modelcfg: ModelsConfig,
    llmModel: llm,
    dataset_name: str,
    img_names: list[str],
    description_paths_by_class: dict[str, list[Path]],
) -> dict[str, Path]:

    """Build one master description per class from current-run descriptions.\n
    The dictionary has object name as key and object description as value and all these object description are concatenated into a list and further sent to different fucntion to summarise them.\n
    The summarised and final version of the object description and objects name
    are combined and written to the master-description file. No historical
    description files are discovered from the output directory.
    """

    missing_classes = [label for label in datacfg.class_labels if not description_paths_by_class.get(label)]
    if missing_classes:
        raise ValueError(
            "No current-run image descriptions were provided for classes: "
            f"{missing_classes}"
        )
    os.makedirs(datacfg.output.master_descriptions_dir,exist_ok=True)
    dataset_directory_name = Path(datacfg.output.master_descriptions_dir) / dataset_name
    os.makedirs(dataset_directory_name,exist_ok= True)
    generated_master_paths: dict[str, Path] = {}
    for class_directory in datacfg.class_labels:
        master_description_class_directory_path = dataset_directory_name / class_directory
        os.makedirs(master_description_class_directory_path,exist_ok=True)
        description_files = [Path(path) for path in description_paths_by_class[class_directory]]
        missing_files = [path for path in description_files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Current-run description files do not exist for class {class_directory!r}: {missing_files}"
            )
        logger.debug(
            "Master description inputs for class %r (%d current-run files): %s",
            class_directory,
            len(description_files),
            description_files,
        )
        object_description_list = []
        for image_description_path in description_files:
                with image_description_path.open('r', encoding='utf-8') as file:
                # Load the JSON data into a Python dictionary or list
                    data = json.load(file)
                    if not isinstance(data, str):
                        raise TypeError(f"Image description must be a JSON string: {image_description_path}")
                    logger.debug("Processing image description: %s", image_description_path.name)
                    logger.debug("Raw description data: %s", data)
                    object_description = {}
                    for item in img_names:
                        description_block = _extract_object_description(data, item)
                        if description_block is None:
                            logger.warning(
                                "Object %r was not found in description file %s; skipping it for this image.",
                                item,
                                image_description_path,
                            )
                            continue
                        object_description[item] = description_block
                    
                    # if not object_description:
                    #     print(f"[WARN] No objects matched at all in '{img_desc_path}'. "
                    #     f"Check that img_names matches the actual object headers in the description.")

                    if object_description:
                        object_description_list.append(object_description)
        if not object_description_list:
            raise ValueError(
                f"None of the configured object labels {img_names!r} matched the current-run "
                f"descriptions for class {class_directory!r}. "
                "Check the normal_descriptor.j2 output format and the generated description files."
            )
        class_wise_object_description = object_description_processing(object_description_list)
        logger.debug("Class '%s' object description: %s", class_directory, class_wise_object_description)
        generated_master_paths[class_directory] = master_description_file_writer(
            datacfg=datacfg,
            modelcfg=modelcfg,
            llmModel=llmModel,
            object_description=class_wise_object_description,
            class_directory_path=master_description_class_directory_path,
            class_label=class_directory,
        )
    return generated_master_paths
