from neo4j import AsyncGraphDatabase, AsyncDriver
from loguru import logger

from config.settings import settings


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )
        logger.info("Neo4j driver initialised ({})", settings.NEO4J_URI)

    async def ping(self) -> bool:
        if self._driver is None:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning("Neo4j ping failed: {}", e)
            return False

    async def run(self, query: str, params: dict | None = None) -> list[dict]:
        if self._driver is None:
            raise RuntimeError("Neo4j driver not initialised — call connect() first")
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [record.data() async for record in result]

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")


neo4j_client = Neo4jClient()
