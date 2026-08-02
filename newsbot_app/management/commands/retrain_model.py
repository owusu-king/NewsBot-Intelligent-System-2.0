"""
Management command: forces a full retrain of the classifier/topic model
from current data and saves the result to data/models/, so the next server
start (and the current one, in-memory) uses the freshly trained model.

Usage:
    python manage.py retrain_model
    python manage.py retrain_model --data-path data/raw/my_new_articles.csv
"""
from django.core.management.base import BaseCommand
from newsbot_app.nlp_service import NewsBotService, MODEL_PATH, MODEL_META_PATH
import json


class Command(BaseCommand):
    help = "Retrain the NewsBot classifier/topic model from current data and save it to data/models/."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-path", type=str, default=None,
            help="Optional path to a CSV to train on (must have Text/content and Category/category "
                 "columns, or the BBC Kaggle format). Defaults to data/raw/BBC News Train.csv, "
                 "falling back to the bundled data/sample/sample_news.csv.",
        )

    def handle(self, *args, **options):
        self.stdout.write("Retraining NewsBot model - this loads the data and refits the "
                           "classifier, TF-IDF vectorizer, and topic model from scratch...")

        service = NewsBotService.get_instance()
        service.retrain(data_path=options["data_path"])

        self.stdout.write(self.style.SUCCESS(
            f"Retrained on {len(service.df)} articles from: {service.dataset_source}"
        ))
        self.stdout.write(f"Best classifier: {service.classifier.best_model_name}")
        self.stdout.write(f"Saved model to: {MODEL_PATH}")

        if MODEL_META_PATH.exists():
            meta = json.loads(MODEL_META_PATH.read_text())
            self.stdout.write(f"Model metadata: {json.dumps(meta, indent=2)}")

        self.stdout.write(self.style.WARNING(
            "Note: this process's in-memory model is now updated, but other already-running "
            "server processes/workers will keep using their old in-memory model until they "
            "restart (they will then automatically load the newly saved file)."
        ))
