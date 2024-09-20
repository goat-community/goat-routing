import asyncio

import sentry_sdk
from celery import Celery, signals
from redis import Redis

from src.core.config import settings
from src.crud.crud_catchment_area import CRUDCatchmentArea
from src.db.session import async_session

celery_app = Celery("worker", broker=settings.CELERY_BROKER_URL)
redis = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
)
crud_catchment_area = CRUDCatchmentArea(async_session(), redis)


@signals.celeryd_init.connect
def init_sentry(**_kwargs):
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=1.0 if settings.ENVIRONMENT == "prod" else 0.1,
    )


@celery_app.task
def run_catchment_area(params):
    loop = asyncio.get_event_loop()
    coroutine = crud_catchment_area.run(params)
    loop.run_until_complete(coroutine)
    return "OK"
