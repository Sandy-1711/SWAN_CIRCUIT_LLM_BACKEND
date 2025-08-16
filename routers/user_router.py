from fastapi import APIRouter, Depends, Query, Body, Request
from fastapi.responses import StreamingResponse
import json
from dependencies import get_current_user, get_db
from typing import Optional
from sqlalchemy.orm import Session
from model import RoleEnum
from schemas import (
    UserUpdate,
    UserOut,
    NewChat,
    CurrentChat,
    FeedbackChunk,
)
from utils.convert_to_ir import convert_to_ir
from rag.run import ingest_feedback_chunk

from crud import update_user, update_message, get_chat_by_id, get_all_chats_by_userid
from services.user_service import create_new_chat

from services.model_service import generate_response_stream
from crud import create_message

router = APIRouter()


@router.get("/me", response_model=UserOut)
async def read_profile(current_user=Depends(get_current_user)):
    return current_user


@router.put("/me", response_model=UserOut)
async def update_profile(
    update: UserUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return update_user(db, current_user.id, update)


@router.get("/chat")
async def get_chat(
    id: int = Query(...), user=Depends(get_current_user), db: Session = Depends(get_db)
):
    chat = get_chat_by_id(db, id, user.id)
    return {
        "id": chat.id,
        "chat_name": chat.chat_name,
        "messages": [
            {
                "model": m.model_id,
                "id": m.id,
                "role": m.role.value,
                "message": m.message,
                "code": m.code,
                "prompt": m.prompt,
                "output": m.output,
                "created_at": m.created_at.isoformat(),
            }
            for m in chat.messages
        ],
    }


@router.get("/all_chats")
async def get_all_chat(user=Depends(get_current_user), db: Session = Depends(get_db)):
    chats = get_all_chats_by_userid(db, user.id)
    return [
        {
            "id": chat.id,
            "chat_name": chat.chat_name,
            "created_at": chat.created_at.isoformat(),
        }
        for chat in chats
    ]


@router.post("/new_chat")
async def handle_new_chat(
    request: Request,
    model: Optional[str] = Query(default="chained"),
    newchat: NewChat = Body(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    async def stream_response():
        # Step 0: send prompt start
        yield json.dumps({"status": "start", "prompt": newchat.prompt}) + "\n"
        # Step 1: naming generator and chat creation
        for chunk in create_new_chat(newchat.prompt, user.id, db):
            print(chunk)
            if await request.is_disconnected():
                print("Client disconnected! Cancelling inference/streaming.")
                return
            yield json.dumps(chunk) + "\n"
            if chunk.get("status") == "saved":
                chat_id = chunk["chat_id"]

        mess = create_message(
            db, user.id, chat_id, message=newchat.prompt, role=RoleEnum.user
        )
        yield (
            json.dumps(
                {
                    "status": "user_message_saved",
                    "message": newchat.prompt,
                    "chat_id": chat_id,
                    "msg_id": mess.id,
                }
            )
            + "\n"
        )
        # Step 2: model response (starts only after chat is saved)
        for chunk in generate_response_stream(
            newchat.prompt, chat_id, user.id, model, db
        ):
            if await request.is_disconnected():
                print("Client disconnected! Cancelling inference/streaming.")
                return
            if chunk.get("status") == "abort":
                yield json.dumps({
                    "status": "error",
                    "reason": chunk.get("reason", "Request aborted by system"),
                    "distance": chunk.get("score", None)
                }) + "\n"
                return

            yield json.dumps(chunk) + "\n"

        yield json.dumps({"status": "done"}) + "\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

    # return {
    #     "prompt": newchat.prompt,
    #     "response": llm_response.get("response"),
    #     "code": llm_response.get("code"),
    #     "output": llm_response.get("output"),
    #     "chat_id": new_chat_created.id,
    #     "chat_name": new_chat_created.chat_name,
    #     "user_id": user.id,
    # }


# @router.post("/chat")
# async def handle_chat(
#     current_chat: CurrentChat,
#     user=Depends(get_current_user),
#     model: Optional[str] = Query(
#         default="chained", description="Model to use (e.g., chained, baseline, graph)"
#     ),
#     db: Session = Depends(get_db),
# ):
#     #     return {
#     #         "propmt": current_chat.prompt,
#     #         "response": "Here is the generated code and diagram",
#     #         "code": """#include <Adafruit_Sensor.h>
#     # #include <Wire.h>
#     # #include <PID_v1.h>

#     # #define MOTEUR1_PIN 9
#     # #define MOTEUR2_PIN 10
#     # #define MOTEUR3_PIN 11
#     # #define MOTEUR4_PIN 12

#     # PID pidX(0, 0, 0);
#     # PID pidY(0, 0, 0);
#     # PID pidZ(0, 0, 0);

#     # float xSetpoint = 0, ySetpoint = 0, zSetpoint = 0;
#     # float xVal = 0, yVal = 0, zVal = 0;

#     # void setup() {
#     #   pinMode(MOTEUR1_PIN, OUTPUT);
#     #   pinMode(MOTEUR2_PIN, OUTPUT);
#     #   pinMode(MOTEUR3_PIN, OUTPUT);
#     #   pinMode(MOTEUR4_PIN, OUTPUT);

#     #   pidX.SetOutputLimits(-100, 100);
#     #   pidY.SetOutputLimits(-100, 100);
#     #   pidZ.SetOutputLimits(-100, 100);

#     #   pidX.SetTunings(1, 0.1, 1);
#     #   pidY.SetTunings(1, 0.1, 1);
#     #   pidZ.SetTunings(1, 0.1, 1);

#     #   Serial.begin(9600);
#     #   Wire.begin();
#     # }

#     # void loop() {
#     #   xVal = analogRead(A0);
#     #   yVal = analogRead(A1);
#     #   zVal = analogRead(A2);

#     #   pidX.Compute(xSetpoint, xVal);
#     #   pidY.Compute(ySetpoint, yVal);
#     #   pidZ.Compute(zSetpoint, zVal);

#     #   analogWrite(MOTEUR1_PIN, pidX.Output());
#     #   analogWrite(MOTEUR2_PIN, pidY.Output());
#     #   analogWrite(MOTEUR3_PIN, pidZ.Output());
#     #   analogWrite(MOTEUR4_PIN, 0);

#     #   Serial.print("X: "); Serial.print(xVal);
#     #   Serial.print(" Y: "); Serial.print(yVal);
#     #   Serial.print(" Z: "); Serial.print(zVal);
#     #   Serial.print(" | PID X: "); Serial.print(pidX.Output());
#     #   Serial.print(" Y: "); Serial.print(pidY.Output());
#     #   Serial.print(" Z: "); Serial.println(pidZ.Output());
#     #   delay(100);
#     # }""",
#     #         "output": {
#     #             "parts": [
#     #                 {"id": "esp", "type": "wokwi-esp8266"},
#     #                 {"id": "dht", "type": "wokwi-dht11"},
#     #                 {"id": "led1", "type": "wokwi-led", "attrs": {"color": "red"}},
#     #                 {"id": "bb1", "type": "wokwi-breadboard"},
#     #             ],
#     #             "connections": [["esp:3V3", "bb1:tp.36", "red", ["v0"]]],
#     #         },
#     #     }

#     llm_response = await generate_response(
#         current_chat.prompt, current_chat.chat_id, user.id, model, db
#     )
#     return {
#         "prompt": current_chat.prompt,
#         "response": llm_response.get("response"),
#         "code": llm_response.get("code"),
#         "output": llm_response.get("output"),
#     }


@router.post("/chat_stream")
async def handle_chat_stream(
    current_chat: CurrentChat,
    request: Request,
    user=Depends(get_current_user),
    model: Optional[str] = Query(default="chained"),
    db: Session = Depends(get_db),
):
    async def stream_response():
        # Step 0: Send initial message
        yield json.dumps({"status": "start", "prompt": current_chat.prompt}) + "\n"
        created_user_message = create_message(
            db,
            user.id,
            current_chat.chat_id,
            message=current_chat.prompt,
            role=RoleEnum.user,
        )
        yield (
            json.dumps(
                {
                    "status": "user_message_saved",
                    "message": current_chat.prompt,
                    "msg_id": created_user_message.id,
                }
            )
            + "\n"
        )
        msg_id = None
        for chunk in generate_response_stream(
            current_chat.prompt, current_chat.chat_id, user.id, model, db
        ):
            if await request.is_disconnected():
                print("Client disconnected! Cancelling inference/streaming.")
                return
            print(chunk)
            if chunk.get("status") == "abort":
                yield json.dumps({
                    "status": "error",
                    "reason": chunk.get("reason", "Request aborted by system"),
                    "distance": chunk.get("score", None)
                }) + "\n"
                return

            if chunk.get("status") == "saved":
                msg_id = chunk["msg_id"]

            yield json.dumps(chunk) + "\n"

        yield (
            json.dumps(
                {"status": "done", "prompt": current_chat.prompt, "msg_id": msg_id}
            )
            + "\n"
        )

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@router.put("/feedback")
async def handle_save_message(
    chunk: FeedbackChunk,
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prompt = chunk.prompt
    message_id = chunk.msgId
    code = chunk.code
    output = chunk.output
    circuit_space_representation = convert_to_ir(output)

    result = ingest_feedback_chunk(
        {
            "prompt": prompt,
            "code": code,
            "output": output,
            "circuit_space_representation": circuit_space_representation,
        }
    )
    print(result)
    # return result
    # return {
    #     "status": "ok",
    #     "chunk_id": f"chunk_{next_chunk_idx}",
    #     "nodes_added": len(nodes),
    #     "embed_rows_added": new_embeds.shape[0],
    #     "embed_row_span": [start, end],
    # }

    update_message(
        db,
        message_id=message_id,
        prompt=prompt,
        code=code,
        output=json.dumps(output),
    )
    return result
