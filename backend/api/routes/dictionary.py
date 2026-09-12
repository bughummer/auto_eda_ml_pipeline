"""Data dictionary endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from backend.api.dependencies import get_container
from backend.services.dictionary import DataDictionaryService
from ml_engine.contracts.dictionary import DataDictionary

router = APIRouter(prefix="/experiments", tags=["data dictionary"])


def get_dictionary_service(request: Request) -> DataDictionaryService:
    return get_container(request).dictionary


DictionaryServiceDep = Annotated[DataDictionaryService, Depends(get_dictionary_service)]


@router.post("/{experiment_id}/data-dictionary", response_model=DataDictionary)
async def upload_data_dictionary(
    experiment_id: str,
    service: DictionaryServiceDep,
    file: Annotated[UploadFile, File(description="JSON, CSV or XLSX column documentation")],
) -> DataDictionary:
    payload = await file.read()
    return service.upload(experiment_id, filename=file.filename or "dictionary", payload=payload)


@router.get("/{experiment_id}/data-dictionary", response_model=DataDictionary)
def get_data_dictionary(experiment_id: str, service: DictionaryServiceDep) -> DataDictionary:
    return service.get(experiment_id)
