
import re
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DescriptionAnalyzer:

    def __init__(self):

        # description_analyzer.py is inside src/description_mode/
        # Knowledge files remain inside src/knowledge/

        self.base_dir = Path(__file__).resolve().parent.parent
        self.knowledge_dir = self.base_dir / "knowledge"

        self.common = self._load_yaml("common.yaml")
        self.ml = self._load_yaml("ml.yaml")
        self.nlp = self._load_yaml("nlp.yaml")

        self.knowledge = {
            "ml": self.ml,
            "nlp": self.nlp
        }

        # Additional algorithm aliases.
        # YAML aliases are still used as the primary source.

        self.algorithm_aliases = {

            "ml": {

                "logistic_regression": [
                    "logistic",
                    "logistic regression",
                    "logit"
                ],

                "linear_regression": [
                    "linear regression",
                    "linear regressor",
                    "ols",
                    "ordinary least squares"
                ],

                "support_vector_machine": [
                    "svm",
                    "svc",
                    "support vector machine",
                    "support vector classifier"
                ],

                "support_vector_regression": [
                    "svr",
                    "support vector regression"
                ],

                "random_forest": [
                    "random forest",
                    "rf"
                ],

                "decision_tree": [
                    "decision tree",
                    "dt"
                ],

                "gradient_boosting": [
                    "gradient boosting",
                    "gbm"
                ],

                "xgboost": [
                    "xgboost",
                    "xg boost"
                ],

                "knn": [
                    "knn",
                    "k nearest neighbors",
                    "k nearest neighbours"
                ],

                "naive_bayes": [
                    "naive bayes",
                    "nb"
                ]
            },

            "nlp": {

                "logistic_regression": [
                    "logistic",
                    "logistic regression",
                    "logit"
                ],

                "linear_svc": [
                    "svc",
                    "svm",
                    "linear svc",
                    "linear svm",
                    "linear support vector classifier"
                ],

                "multinomial_naive_bayes": [
                    "naive bayes",
                    "multinomial naive bayes",
                    "mnb"
                ],

                "random_forest": [
                    "random forest",
                    "rf"
                ]
            }
        }

        self.spelling_corrections = {
            "predicition": "prediction",
            "prediciton": "prediction",
            "predicit": "predict",
            "classifer": "classifier",
            "classificaton": "classification",
            "regresion": "regression",
            "regresssion": "regression"
        }

    # --------------------------------------------------
    # LOAD KNOWLEDGE BASE
    # --------------------------------------------------

    def _load_yaml(self, filename):

        path = self.knowledge_dir / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Knowledge file not found: {path}"
            )

        with open(path, "r", encoding="utf-8") as file:

            data = yaml.safe_load(file)

        if not isinstance(data, dict):

            raise ValueError(
                f"Invalid or empty knowledge file: {filename}"
            )

        return data

    # --------------------------------------------------
    # TEXT PROCESSING
    # --------------------------------------------------

    def normalize(self, text):

        text = str(text).lower()

        for wrong, correct in self.spelling_corrections.items():

            text = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                text
            )

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        return re.sub(
            r"\s+",
            " ",
            text
        ).strip()

    @staticmethod
    def contains_phrase(text, phrase):

        if not phrase:
            return False

        pattern = (
            r"\b"
            + re.escape(phrase.lower())
            + r"\b"
        )

        return re.search(
            pattern,
            text.lower()
        ) is not None

    def contains_any(self, text, phrases):

        return any(
            self.contains_phrase(text, phrase)
            for phrase in phrases
        )

    # --------------------------------------------------
    # PROJECT NAME
    # --------------------------------------------------

    def extract_project_name(self, description):

        text = self.normalize(description)

        # Remove introductory phrases.

        text = re.sub(
            r"^(build|create|develop|make|design|implement)\s+",
            "",
            text
        )

        text = re.sub(
            r"^(an|a|the)\s+",
            "",
            text
        )

        # Remove implementation details.

        stop_phrases = [
            " using ",
            " with ",
            " that ",
            " where ",
            " which ",
            " requiring ",
            " require ",
            " and require ",
            " and include ",
            " and build ",
            " only ",
            " streamlit ",
            " flask ",
            " fastapi ",
            " using only "
        ]

        cut_positions = []

        for phrase in stop_phrases:

            position = text.find(phrase)

            if position >= 0:
                cut_positions.append(position)

        if cut_positions:
            text = text[:min(cut_positions)]

        # Remove generic words.

        text = re.sub(
            r"\b(system|application|website|web app|model|project)\b",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if not text:
            return "Generated_ML_Project"

        words = text.split()

        name = "_".join(
            word.capitalize()
            for word in words[:6]
        )

        return name[:60].strip("_")

    # --------------------------------------------------
    # DOMAIN DETECTION
    # --------------------------------------------------

    def detect_domain(self, description):

        text = self.normalize(description)

        # Explicit NLP indicators.

        nlp_terms = [

            "natural language processing",
            "nlp",

            "text data",
            "text classification",
            "text classifier",
            "text regression",
            "text prediction",
            "text clustering",

            "sentiment analysis",
            "sentiment classification",
            "sentiment prediction",

            "document classification",
            "document classifier",
            "document clustering",

            "named entity recognition",
            "entity extraction",
            "ner",

            "keyword extraction",
            "text summarization",
            "text generation",

            "language model",
            "word embeddings",
            "sentence embeddings",

            "customer reviews",
            "product reviews",
            "movie reviews",
            "user reviews",

            "imdb",
            "news articles",
            "news classification",
            "spam emails",
            "email classification",

            "text messages",
            "tweets",
            "social media posts"
        ]

        if self.contains_any(text, nlp_terms):
            return "nlp"

        # ML indicators.

        ml_terms = [

            "tabular data",
            "structured data",
            "numerical features",
            "customer dataset",
            "sales data",
            "financial data",
            "sensor data",
            "medical records",
            "patient data",
            "transaction data",
            "time series data",
            "numerical prediction"
        ]

        if self.contains_any(text, ml_terms):
            return "ml"

        # Default domain.

        return "ml"

    # --------------------------------------------------
    # TASK DETECTION
    # --------------------------------------------------

    def match_task(self, description, domain):

        data = self.knowledge[domain]

        tasks = data.get("tasks", {})

        text = self.normalize(description)

        matches = []

        # 1. Match YAML aliases.

        for task_id, info in tasks.items():

            aliases = info.get("aliases", [])

            score = 0
            matched_terms = []

            for alias in aliases:

                alias = self.normalize(str(alias))

                if alias and self.contains_phrase(text, alias):

                    score += len(alias.split()) + 1

                    matched_terms.append(alias)

            if score:

                matches.append({

                    "task": task_id,
                    "score": score,
                    "matched_terms": matched_terms

                })

        if matches:

            return max(
                matches,
                key=lambda item: item["score"]
            )

        # 2. Generic task detection.

        if domain == "nlp":

            task = self.detect_nlp_task(text, tasks)

        else:

            task = self.detect_ml_task(text, tasks)

        if task:
            return task

        # 3. TF-IDF fallback.

        documents = []

        task_ids = list(tasks.keys())

        for task_id in task_ids:

            info = tasks[task_id]

            document = " ".join([

                task_id.replace("_", " "),

                str(info.get("description", "")),

                " ".join(
                    str(alias)
                    for alias in info.get("aliases", [])
                )

            ])

            documents.append(document)

        try:

            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2)
            )

            vectors = vectorizer.fit_transform(
                [text] + documents
            )

            similarities = cosine_similarity(
                vectors[0:1],
                vectors[1:]
            )[0]

            index = similarities.argmax()

            if similarities[index] >= 0.15:

                return {

                    "task": task_ids[index],

                    "score": float(similarities[index]),

                    "matched_terms": []

                }

        except ValueError:
            pass

        return None

    # --------------------------------------------------
    # GENERIC ML TASK DETECTION
    # --------------------------------------------------

    def detect_ml_task(self, text, tasks):

        # Classification indicators.

        classification_terms = [

            "classification",
            "classifier",
            "classify",
            "categorize",
            "categorise",
            "categorization",
            "categorisation",
            "predict category",
            "predict a category",
            "predict class",
            "predict a class",
            "identify category",
            "identify class",
            "binary classification",
            "multiclass classification",
            "multi class classification",
            "label prediction",
            "spam detection",
            "fraud detection",
            "disease detection",
            "disease diagnosis",
            "churn prediction"

        ]

        # Regression indicators.

        regression_terms = [

            "regression",
            "regressor",
            "regression model",
            "regression analysis",
            "predict a value",
            "predict value",
            "predict numerical",
            "predict numeric",
            "estimate a value",
            "estimate value",
            "continuous value",
            "continuous variable",
            "numerical prediction",
            "numeric prediction",
            "price prediction",
            "value prediction",
            "salary prediction",
            "score prediction",
            "sales forecasting",
            "demand forecasting",
            "forecast a value",
            "predict quantity",
            "estimate quantity"

        ]

        # Clustering indicators.

        clustering_terms = [

            "clustering",
            "cluster data",
            "cluster records",
            "cluster customers",
            "group similar",
            "group records",
            "group customers",
            "discover groups",
            "find groups",
            "customer segmentation",
            "data segmentation",
            "unsupervised grouping"

        ]

        # Anomaly detection indicators.

        anomaly_terms = [

            "anomaly detection",
            "detect anomalies",
            "outlier detection",
            "detect outliers",
            "identify anomalies",
            "identify outliers",
            "unusual behavior",
            "unusual behaviour",
            "fraudulent transactions",
            "abnormal activity"

        ]

        # Apply task detection.

        if self.contains_any(text, anomaly_terms):

            if "anomaly_detection" in tasks:

                return {
                    "task": "anomaly_detection",
                    "score": 1.0,
                    "matched_terms": ["anomaly detection"]
                }

        if self.contains_any(text, clustering_terms):

            if "clustering" in tasks:

                return {
                    "task": "clustering",
                    "score": 1.0,
                    "matched_terms": ["clustering"]
                }

        if self.contains_any(text, classification_terms):

            if "classification" in tasks:

                return {
                    "task": "classification",
                    "score": 1.0,
                    "matched_terms": ["classification"]
                }

        if self.contains_any(text, regression_terms):

            if "regression" in tasks:

                return {
                    "task": "regression",
                    "score": 1.0,
                    "matched_terms": ["regression"]
                }

        return None

    # --------------------------------------------------
    # GENERIC NLP TASK DETECTION
    # --------------------------------------------------

    def detect_nlp_task(self, text, tasks):

        # Named entity recognition.

        ner_terms = [

            "named entity recognition",
            "entity extraction",
            "extract entities",
            "extract names",
            "identify named entities",
            "ner"

        ]

        if self.contains_any(text, ner_terms):

            if "named_entity_recognition" in tasks:

                return {
                    "task": "named_entity_recognition",
                    "score": 1.0,
                    "matched_terms": ["named entity recognition"]
                }

        # Sentiment analysis.

        sentiment_terms = [

            "sentiment analysis",
            "sentiment classification",
            "sentiment prediction",
            "sentiment detection",
            "opinion mining",
            "emotion classification",
            "emotion detection",
            "polarity detection",
            "positive negative classification",
            "review sentiment"

        ]

        if self.contains_any(text, sentiment_terms):

            if "sentiment_analysis" in tasks:

                return {
                    "task": "sentiment_analysis",
                    "score": 1.0,
                    "matched_terms": ["sentiment analysis"]
                }

        # Text regression.

        text_regression_terms = [

            "text regression",
            "text score prediction",
            "predict score from text",
            "predict a score from text",
            "estimate score from text",
            "rating prediction from text",
            "predict rating from text",
            "continuous value from text",
            "numerical value from text"

        ]

        if self.contains_any(text, text_regression_terms):

            if "text_regression" in tasks:

                return {
                    "task": "text_regression",
                    "score": 1.0,
                    "matched_terms": ["text regression"]
                }

        # Text clustering.

        text_clustering_terms = [

            "text clustering",
            "document clustering",
            "cluster documents",
            "cluster text",
            "group similar documents",
            "group similar texts",
            "group similar articles",
            "group similar messages"

        ]

        if self.contains_any(text, text_clustering_terms):

            if "text_clustering" in tasks:

                return {
                    "task": "text_clustering",
                    "score": 1.0,
                    "matched_terms": ["text clustering"]
                }

        # Keyword extraction.

        keyword_terms = [

            "keyword extraction",
            "extract keywords",
            "identify keywords",
            "important keywords",
            "key phrases",
            "phrase extraction"

        ]

        if self.contains_any(text, keyword_terms):

            if "keyword_extraction" in tasks:

                return {
                    "task": "keyword_extraction",
                    "score": 1.0,
                    "matched_terms": ["keyword extraction"]
                }

        # Text summarization.

        summarization_terms = [

            "text summarization",
            "document summarization",
            "summarize text",
            "summarize documents",
            "generate summary",
            "generate summaries",
            "automatic summarization"

        ]

        if self.contains_any(text, summarization_terms):

            if "text_summarization" in tasks:

                return {
                    "task": "text_summarization",
                    "score": 1.0,
                    "matched_terms": ["text summarization"]
                }

        # General text classification.

        classification_terms = [

            "text classification",
            "text classifier",
            "document classification",
            "document classifier",
            "classify text",
            "classify documents",
            "classify messages",
            "classify reviews",
            "classify articles",
            "email classification",
            "email classifier",
            "spam classifier",
            "news classifier",
            "imdb classifier",
            "classifier",
            "classification",
            "categorize text",
            "categorise text",
            "categorize documents",
            "categorise documents"

        ]

        if self.contains_any(text, classification_terms):

            if "text_classification" in tasks:

                return {
                    "task": "text_classification",
                    "score": 1.0,
                    "matched_terms": ["text classification"]
                }

        return None

    # --------------------------------------------------
    # ALGORITHM DETECTION
    # --------------------------------------------------

    def detect_algorithms(self, description, domain):

        data = self.knowledge[domain]

        algorithms = data.get("algorithms", {})

        text = self.normalize(description)

        selected = []

        extra_aliases = self.algorithm_aliases.get(
            domain,
            {}
        )

        for algorithm_id, info in algorithms.items():

            aliases = info.get("aliases", [])

            display_name = info.get(
                "display_name",
                algorithm_id
            )

            names = (
                list(aliases)
                + [display_name]
                + extra_aliases.get(algorithm_id, [])
            )

            matched = False

            for name in names:

                normalized_name = self.normalize(str(name))

                if normalized_name and self.contains_phrase(
                    text,
                    normalized_name
                ):

                    matched = True
                    break

            if matched:

                selected.append({

                    "id": algorithm_id,

                    "name": display_name,

                    "selection_type": "user_specified"

                })

        return selected

    # --------------------------------------------------
    # AUTOMATIC ALGORITHM RECOMMENDATIONS
    # --------------------------------------------------

    def recommend_algorithms(self, domain, task):

        data = self.knowledge[domain]

        recommendations = []

        for algorithm_id, info in data.get(
            "algorithms", {}
        ).items():

            if task in info.get("supported_tasks", []):

                recommendations.append({

                    "id": algorithm_id,

                    "name": info.get(
                        "display_name",
                        algorithm_id
                    ),

                    "selection_type": "recommended",

                    "strengths": info.get("strengths", []),

                    "limitations": info.get("limitations", [])

                })

        return recommendations

    # --------------------------------------------------
    # INTERFACE DETECTION
    # --------------------------------------------------

    def detect_interface(self, description):

        text = self.normalize(description)

        if self.contains_phrase(text, "fastapi"):
            return "fastapi"

        if self.contains_phrase(text, "flask"):
            return "flask"

        if self.contains_phrase(text, "streamlit"):
            return "streamlit"

        if self.contains_any(text, [

            "no interface",
            "without interface",
            "without a website",
            "cli only",
            "command line only"

        ]):

            return "none"

        return "streamlit"

    # --------------------------------------------------
    # DATASET DETECTION
    # --------------------------------------------------

    def detect_dataset(self, description):

        text = self.normalize(description)

        known_datasets = {

            "imdb": "IMDb",
            "mnist": "MNIST",
            "iris": "Iris",
            "titanic": "Titanic",
            "housing": "Housing Dataset"

        }

        for keyword, dataset_name in known_datasets.items():

            if self.contains_phrase(text, keyword):

                return {

                    "name": dataset_name,

                    "provided_by_user": False

                }

        if self.contains_any(text, [

            "my dataset",
            "uploaded dataset",
            "provided dataset",
            "csv file",
            "uploaded csv",
            "input dataset"

        ]):

            return {

                "name": None,

                "provided_by_user": True

            }

        return {

            "name": None,

            "provided_by_user": False

        }

    # --------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------

    def get_preprocessing(self, domain, description):

        data = self.knowledge[domain]

        preprocessing = data.get("preprocessing", {})

        text = self.normalize(description)

        selected = []

        for operation_id, info in preprocessing.items():

            aliases = info.get("aliases", [])

            if any(
                self.contains_phrase(text, str(alias))
                for alias in aliases
            ):

                selected.append(operation_id)

        if not selected:

            if domain == "nlp":

                selected = [
                    "text_cleaning",
                    "tokenization"
                ]

            else:

                selected = [
                    "missing_value_handling",
                    "categorical_encoding"
                ]

        return selected

    # --------------------------------------------------
    # REPRESENTATION DETECTION
    # --------------------------------------------------

    def detect_representations(self, description, domain):

        if domain != "nlp":
            return []

        representations = self.nlp.get(
            "representations", {}
        )

        text = self.normalize(description)

        selected = []

        for representation_id, info in representations.items():

            names = [

                representation_id.replace("_", " "),

                info.get("display_name", "")

            ] + info.get("aliases", [])

            if any(

                self.contains_phrase(text, str(name))

                for name in names

                if name

            ):

                selected.append(representation_id)

        if not selected:
            selected = ["tfidf"]

        return selected

    # --------------------------------------------------
    # COMPATIBILITY VALIDATION
    # --------------------------------------------------

    def validate_algorithms(self, algorithms, task, domain):

        valid = []

        errors = []

        domain_data = self.knowledge[domain]

        available_algorithms = domain_data.get(
            "algorithms", {}
        )

        for algorithm in algorithms:

            algorithm_id = algorithm["id"]

            info = available_algorithms.get(algorithm_id)

            if info is None:

                errors.append(

                    f"Unknown algorithm '{algorithm_id}' "
                    f"for domain '{domain}'."

                )

                continue

            supported_tasks = info.get(
                "supported_tasks", []
            )

            if task not in supported_tasks:

                errors.append(

                    f"{info.get('display_name', algorithm_id)} "
                    f"does not support task '{task}'. "
                    f"Supported tasks: {supported_tasks}"

                )

                continue

            valid.append(algorithm)

        return valid, errors

    # --------------------------------------------------
    # MODEL SELECTION CONFIGURATION
    # --------------------------------------------------

    def get_model_selection_config(self, domain, task):

        if domain == "ml" and task == "regression":

            return {

                "enabled": True,

                "strategy": "cross_validation",

                "primary_metric": "rmse",

                "optimization_direction": "minimize",

                "secondary_metrics": [
                    "mae",
                    "r2"
                ],

                "save_best_model": True,

                "compare_models": True,

                "use_test_set_for_selection": False

            }

        if domain == "ml" and task == "classification":

            return {

                "enabled": True,

                "strategy": "cross_validation",

                "primary_metric": "f1",

                "optimization_direction": "maximize",

                "secondary_metrics": [
                    "accuracy",
                    "precision",
                    "recall"
                ],

                "save_best_model": True,

                "compare_models": True,

                "use_test_set_for_selection": False

            }

        return {

            "enabled": False,

            "strategy": None,

            "primary_metric": None,

            "optimization_direction": None,

            "secondary_metrics": [],

            "save_best_model": False,

            "compare_models": False,

            "use_test_set_for_selection": False

        }

    # --------------------------------------------------
    # MAIN ANALYSIS
    # --------------------------------------------------

    def analyze(self, description):

        if not isinstance(description, str) or not description.strip():

            raise ValueError(
                "Please enter a project description."
            )

        description = description.strip()

        # 1. Detect domain.

        domain = self.detect_domain(description)

        # 2. Detect task.

        task_match = self.match_task(
            description,
            domain
        )

        if not task_match:

            return {

                "success": False,

                "needs_clarification": True,

                "message": (

                    "Could not identify the project task. "

                    "Please describe what the model should predict, "

                    "classify, extract, summarize or discover."

                )

            }

        task = task_match["task"]

        # 3. Detect explicitly selected algorithms.

        specified_algorithms = self.detect_algorithms(
            description,
            domain
        )

        # 4. Respect user algorithm selection.

        if specified_algorithms:

            selected_algorithms = specified_algorithms

            algorithm_mode = "user_specified"

        else:

            selected_algorithms = self.recommend_algorithms(
                domain,
                task
            )

            algorithm_mode = "automatic_recommendation"

        # 5. Validate algorithms.

        valid_algorithms, errors = self.validate_algorithms(
            selected_algorithms,
            task,
            domain
        )

        if errors:

            return {

                "success": False,

                "needs_clarification": True,

                "message": "Algorithm compatibility issue.",

                "errors": errors

            }

        if not valid_algorithms:

            return {

                "success": False,

                "needs_clarification": True,

                "message": (

                    "No compatible algorithms were found "

                    "for the selected task."

                )

            }

        # 6. Task details.

        task_info = self.knowledge[domain]["tasks"][task]

        metrics = task_info.get(
            "default_metrics", []
        )

        # 7. Detect interface.

        interface = self.detect_interface(description)

        # 8. Detect dataset.

        dataset = self.detect_dataset(description)

        # 9. Detect preprocessing.

        preprocessing = self.get_preprocessing(
            domain,
            description
        )

        # 10. Detect representations.

        representations = self.detect_representations(
            description,
            domain
        )

        # 11. Extract project name.

        project_name = self.extract_project_name(
            description
        )

        # 12. Model selection configuration.

        model_selection = self.get_model_selection_config(
            domain,
            task
        )

        # 13. Build architecture.

        architecture = {

            "project_name": project_name,

            "domain": domain,

            "task": task,

            "task_description": task_info.get(
                "description", ""
            ),

            "original_description": description,

            "algorithms": valid_algorithms,

            "algorithm_mode": algorithm_mode,

            "preprocessing": preprocessing,

            "representations": representations,

            "evaluation_metrics": metrics,

            "dataset": dataset,

            "interface": interface,

            "model_selection": model_selection,

            "generation": {

                "include_preprocessing": True,

                "include_training": True,

                "include_prediction": True,

                "include_evaluation": True,

                "include_web_app": interface != "none",

                "include_readme": True,

                "include_requirements": True

            },

            "validation": {

                "is_valid": True,

                "errors": []

            }

        }

        return {

            "success": True,

            "needs_clarification": False,

            "architecture": architecture

        }


# --------------------------------------------------
# PUBLIC FUNCTION FOR APP.PY
# --------------------------------------------------

def analyze_description(description):

    analyzer = DescriptionAnalyzer()

    return analyzer.analyze(description)
