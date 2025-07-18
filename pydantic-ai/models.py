from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    data_data_schema: Optional[str] = None

class Press20Data(BaseModel):
    # id
    id: Optional[int] = None
    # fault and part id
    shot_num: Optional[int] = None
    overallPassFail: Optional[str] = None
    dataset_id: Optional[str] = None
    cameratimestamp: Optional[str] = None
    bottomPassFail: Optional[str] = None
    topPassFail: Optional[str] = None
    bottomAnomalyLevel: Optional[float] = None
    topAnomalyLevel: Optional[float] = None
    # machine data 
    machineTimestamp: Optional[str] = None
    actCycleTime: Optional[int] = None
    actClpClsTime: Optional[int] = None
    actCoolingTime: Optional[int] = None
    actCurrentServodrive_Disp_1: Optional[int] = None
    actCurrentServodrive_Disp_2: Optional[int] = None
    actCurrentServodrive_1: Optional[int] = None
    actCurrentServodrive_2: Optional[int] = None
    actCushionPosition: Optional[int] = None
    actFeedTemp: Optional[int] = None
    actFill: Optional[int] = None
    actFillTime_0: Optional[int] = None
    actFillTime_1: Optional[int] = None
    actFillTime_2: Optional[int] = None
    actInjectionPos: Optional[int] = None
    actInjFillSpd: Optional[int] = None
    actCalEjtFwdSpd: Optional[int] = None
    actCalEjtRetSpd: Optional[int] = None
    actInjFwdStagePos_0: Optional[int] = None
    actInjFwdStagePos_1: Optional[int] = None
    actInjFwdStagePos_2: Optional[int] = None
    inj_Act_Prs_0: Optional[int] = None
    inj_Act_Prs_1: Optional[int] = None
    inj_Act_Prs_2: Optional[int] = None
    actInjFwdStagePrs_0: Optional[int] = None
    actInjFwdStagePrs_1: Optional[int] = None
    actInjFwdStagePrs_2: Optional[int] = None
    actInjFwdStageTime_0: Optional[int] = None
    actInjFwdStageTime_1: Optional[int] = None
    actInjFwdStageTime_2: Optional[int] = None
    actMotorRPMServodrive_0: Optional[int] = None
    actMotorRPMServodrive_1: Optional[int] = None
    actNozzleCurrent: Optional[int] = None
    actNozzlePIDPer: Optional[int] = None
    actNozzleTemp: Optional[int] = None
    actoiltemp: Optional[int] = None
    actProcOfMaxInjPrs: Optional[int] = None
    actProcOfMaxInjPrsPos: Optional[int] = None
    actRearSuckbackSpd: Optional[int] = None
    actRearSuckbackTime: Optional[int] = None
    actRefillTime: Optional[int] = None
    actSysPrsServodrive_0: Optional[int] = None
    actSysPrsServodrive_1: Optional[int] = None
    actTempServodrive_0: Optional[int] = None
    actTempServodrive_1: Optional[int] = None
    actTempServoMotor_0: Optional[int] = None
    actTempServoMotor_1: Optional[int] = None
    actCCprs: Optional[int] = None
    actZone1Temp: Optional[int] = None
    actZone2Temp: Optional[int] = None
    actZone3Temp: Optional[int] = None
    actZone4Temp: Optional[int] = None
    actZone5Temp: Optional[int] = None
    actZone6Temp: Optional[int] = None
    prvActInj1PlastTime: Optional[int] = None
    backprs_value: Optional[int] = None
    actProcMonMinInjPos: Optional[int] = None
    flow_value: Optional[int] = None

class DocumentRow(BaseModel):
    id: Optional[int] = None
    dataset_id: str
    row_data: dict

class QueryInput(BaseModel):
    chatInput: str
    sessionId: str

class ChatHistory(BaseModel):
    id: Optional[int] = None
    userId: Optional[str] = None
    sessionId: Optional[str] = None
    message: dict

class ChatMessage(BaseModel):
    role: str
    content: str
