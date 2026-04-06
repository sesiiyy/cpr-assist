"""Drop the entire MongoDB database named in app settings (all collections)."""

from pymongo import MongoClient

from app.core.config import settings


def main() -> None:
    MongoClient(settings.mongodb_uri).drop_database(settings.mongodb_db)
    print(f"Dropped database {settings.mongodb_db!r}.")


if __name__ == "__main__":
    main()
