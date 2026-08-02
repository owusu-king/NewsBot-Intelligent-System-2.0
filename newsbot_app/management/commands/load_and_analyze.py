"""
Management command: loads the dataset, runs the full NLP pipeline, and
stores results in the database. Useful for pre-populating the dashboard.

Usage: python manage.py load_and_analyze
"""
from django.core.management.base import BaseCommand
from newsbot_app.nlp_service import NewsBotService
from newsbot_app.models import AnalyzedArticle


class Command(BaseCommand):
    help = "Load the news dataset, run the NLP pipeline, and store results in the database."

    def handle(self, *args, **options):
        service = NewsBotService.get_instance()
        self.stdout.write(f"Using dataset: {service.dataset_source}")
        created = 0
        for _, row in service.df.iterrows():
            pred = service.classifier.predict_with_confidence(
                service.feature_extractor.transform([row["content_processed"]])
            )[0]
            AnalyzedArticle.objects.update_or_create(
                source_article_id=row["article_id"],
                defaults=dict(
                    title=row["title"][:500],
                    content=row["content"],
                    category=pred["label"],
                    category_confidence=pred["confidence"],
                    sentiment_label=row["sentiment_label"],
                    sentiment_compound=row["sentiment_compound"],
                    topic_id=int(row["topic_id"]),
                ),
            )
            created += 1
        self.stdout.write(self.style.SUCCESS(f"Analyzed and stored {created} articles."))
