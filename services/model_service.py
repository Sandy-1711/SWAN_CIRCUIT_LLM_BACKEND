from dotenv import load_dotenv
from crud import create_message
from database import SessionLocal
from sqlalchemy.orm import Session

# from inferences.chained_gguf_inference import run_chained_gguf_pipeline
from pipelines.rag_pipeline import run_rag_pipeline
from inferences.baseline_inference import run_baseline_pipeline
from model import RoleEnum
from inferences.chained_stream_inference import run_chained_stream_inference
import json
from tabulate import tabulate

load_dotenv()


def json_to_table_and_summary(json_output: dict) -> dict:
    if (
        not json_output
        or "connections" not in json_output
        or "parts" not in json_output
    ):
        return {"table": "", "summary": ""}

    # Map part IDs to names
    parts_map = {p["id"]: p["type"] for p in json_output["parts"]}

    table = []
    for conn in json_output["connections"]:
        src, dst, color, _ = conn
        src_id, src_pin = src.split(":")
        dst_id, dst_pin = dst.split(":")
        table.append(
            [
                f"{parts_map.get(src_id, src_id)} ({src_id}) pin {src_pin}",
                f"{parts_map.get(dst_id, dst_id)} ({dst_id}) pin {dst_pin}",
                color,
            ]
        )

    # Generate a nice table
    table_str = tabulate(table, headers=["From", "To", "Wire Color"], tablefmt="grid")

    # Generate a natural language summary
    summary_lines = [
        f"The {row[0]} is connected to {row[1]} with a {row[2]} wire." for row in table
    ]
    summary = "\n".join(summary_lines)

    return {"table": table_str, "summary": summary}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# async def generate_response(
#     prompt: str, chat_id: int, user_id: int, model, db: Session
# ) -> dict:
#     code = ""
#     json_output = {}
#     ir = ""

#     if model == "baseline":
#         code, json_output = run_baseline_pipeline(prompt)
#     elif model == "rag":
#         code, ir, json_output = await run_rag_pipeline(prompt)
#     elif model == "chained":
#         code, ir, json_output = run_chained_gguf_pipeline(prompt)

#     print(code)

#     if ir != "":
#         print(ir)

#     print(json_output)

#     created_user_message = create_message(
#         db, user_id, chat_id, message=prompt, role=RoleEnum.user
#     )
#     created_message = create_message(
#         db,
#         user_id,
#         chat_id,
#         model_id=model,
#         prompt=prompt,
#         message="Here is the generated code and diagram",
#         role=RoleEnum.assistant,
#         code=code,
#         output=json_output,
#     )

#     return {
#         "message": created_message.message,
#         "code": created_message.code,
#         "id": created_message.id,
#         "prompt": created_message.prompt,
#         "output": created_message.output,
#     }


def generate_response_stream(
    prompt: str, chat_id: int, user_id: int, model, db: Session
):
    code = ""
    json_output = {}
    ir = ""
    rag_stage_1 = False
    rag_stage_2 = False
    rag_stage_3 = False

    if model == "chained":
        for chunk in run_chained_stream_inference(prompt):
            yield chunk
            if chunk.get("stage") == "code_done":
                code = chunk["code"]
            elif chunk.get("stage") == "ir_done":
                ir = chunk["ir"]
            elif chunk.get("stage") == "json_done":
                json_output = chunk["output"]
                # nl_output = json_to_table_and_summary(json_output)
                # yield {
                #     "stage": "nl_summary",
                #     "table": nl_output["table"],
                #     "summary": nl_output["summary"]
                # }
    elif model == "baseline":
        code, json_output = run_baseline_pipeline(prompt)
        # Fake streaming output to match the rest of the models
        yield {"stage": "code_done", "code": code}
        yield {"stage": "json_done", "output": json_output}
        # nl_output = json_to_table_and_summary(json_output)
        # yield {"stage": "nl_summary", **nl_output}

    elif model == "rag":
        for chunk in run_rag_pipeline(prompt):
            if chunk.get("status") == "abort":
                yield chunk
                return
            yield chunk
            if chunk.get("stage") == "code_done":
                code = chunk["code"]
            elif chunk.get("stage") == "ir_done":
                ir = chunk["ir"]
            elif chunk.get("stage") == "json_done":
                json_output = chunk["output"]
                # nl_output = json_to_table_and_summary(json_output)
                # yield {"stage": "nl_summary", **nl_output}
            elif chunk.get("stage") == "rag_stage_1_done":
                rag_stage_1 = True
            elif chunk.get("stage") == "rag_stage_2_done":
                rag_stage_2 = True
            elif chunk.get("stage") == "rag_stage_3_done":
                rag_stage_3 = True
    # Save to DB
    if not (model == "rag" and chunk.get("status") == "abort"):
        assistant_msg = create_message(
            db,
            user_id,
            chat_id,
            model_id=model,
            prompt=prompt,
            message="Here is the generated code and diagram",
            role=RoleEnum.assistant,
            code=code,
            output=json_output,
        )

        yield {
            "status": "saved",
            "msg_id": assistant_msg.id,
            "final_message": assistant_msg.message,
        }
