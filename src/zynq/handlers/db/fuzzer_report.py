import logging

import motor.motor_asyncio

from zynq.consts import DATABASE_NAME, FUZZER_REPORT_COLLECTION_NAME
from zynq.db.mongodb import MongoDBHandler
from zynq.models.fuzzer_result import FuzzerResult

logger = logging.getLogger(__name__)

class FuzzerReportDBHandler(MongoDBHandler[FuzzerResult]):
    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient, database: str = DATABASE_NAME) -> None:
        super().__init__(client, database, FUZZER_REPORT_COLLECTION_NAME, FuzzerResult)
