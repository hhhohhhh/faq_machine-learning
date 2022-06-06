TEXT = "text"
INTENT = "intent"

# 新增 OUTPUT_INTENT 为大意图字段
OUTPUT_INTENT = 'output_intent'
INTENT_TO_OUTPUT_MAPPING_FILENAME = "intent_to_output_mapping"


RESPONSE = "response"
RESPONSE_SELECTOR = "response_selector"
# retrieval intent with the response key suffix attached : 如 'faq/如何查询自助开户进度'
INTENT_RESPONSE_KEY = "intent_response_key"
ACTION_TEXT = "action_text"
ACTION_NAME = "action_name"
INTENT_NAME_KEY = "name"
FULL_RETRIEVAL_INTENT_NAME_KEY = "full_retrieval_intent_name"
METADATA = "metadata"
METADATA_INTENT = "intent"
METADATA_EXAMPLE = "example"
INTENT_RANKING_KEY = "intent_ranking"
PREDICTED_CONFIDENCE_KEY = "confidence"

RESPONSE_IDENTIFIER_DELIMITER = "/"

FEATURE_TYPE_SENTENCE = "sentence"
FEATURE_TYPE_SEQUENCE = "sequence"
VALID_FEATURE_TYPES = [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]

EXTRACTOR = "extractor"
PRETRAINED_EXTRACTORS = {
    "DucklingEntityExtractor",
    "SpacyEntityExtractor",
    "DucklingHTTPExtractor",  # for backwards compatibility when dumping Markdown
}
TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}

ENTITIES = "entities"
ENTITY_TAGS = "entity_tags"
ENTITY_ATTRIBUTE_TYPE = "entity"
ENTITY_ATTRIBUTE_GROUP = "group"
ENTITY_ATTRIBUTE_ROLE = "role"
ENTITY_ATTRIBUTE_VALUE = "value"
ENTITY_ATTRIBUTE_START = "start"
ENTITY_ATTRIBUTE_END = "end"
NO_ENTITY_TAG = "O"
SPLIT_ENTITIES_BY_COMMA = "split_entities_by_comma"
SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE = True
SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET = {".", ",", " ", ";"}
