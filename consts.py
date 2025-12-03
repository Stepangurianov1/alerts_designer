from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

CSD_DB_HOST = "138.68.88.175"
CSD_DB_PORT = 5432
CSD_DB_NAME = "csd_bi"
CSD_DB_USER = os.getenv("CSD_DB_USER")
CSD_DB_PASSWORD = os.getenv("CSD_DB_PASSWORD")

DWH_DB_HOST = "49.12.21.243"
DWH_DB_PORT = 6432
DWH_DB_NAME = "postgres"
DWH_DB_USER = os.getenv("DWH_DB_USER")
DWH_DB_PASSWORD = os.getenv("DWH_DB_PASSWORD")

CSD_ENGINE = create_engine(
    f"postgresql://{CSD_DB_USER}:{CSD_DB_PASSWORD}"
    f"@{CSD_DB_HOST}:{CSD_DB_PORT}/{CSD_DB_NAME}",
    pool_pre_ping=True,  
)

DWH_ENGINE = create_engine(
    f"postgresql://{DWH_DB_USER}:{DWH_DB_PASSWORD}"
    f"@{DWH_DB_HOST}:{DWH_DB_PORT}/{DWH_DB_NAME}",
    pool_pre_ping=True,
)