# TKNT Backend

FastAPI backend package for the current TKNT pipeline, inventory, auth, account, and snapshot image APIs.

## Run With Docker

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY or the provider keys you use
docker compose up --build
```

Runtime inventory, design knowledge, and size profiles are loaded from Postgres. `TKNT_AUTO_LOAD_DEMO_DATA` defaults to `0`; set it to `1` only when you intentionally want to seed the bundled `synthetic_data/inventory.json` into Postgres before serving real requests.

## Switch OpenAI And Azure

Use OpenAI:

```dotenv
OPENAI_AZURE=0
OPENAI_API_KEY=sk-...
OPENAI_PRIMARY_MODEL=gpt-5.4-mini
OPENAI_HELPER_MODEL=gpt-5.4-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use Azure OpenAI:

```dotenv
OPENAI_AZURE=1
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com
AZURE_OPENAI_API_KEY=<azure-key>
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
```

For Azure, the `*_DEPLOYMENT` values are Azure deployment names, not model catalog names. `AZURE_OPENAI_PRIMARY_DEPLOYMENT` and `AZURE_OPENAI_HELPER_DEPLOYMENT` can override the shared chat deployment when needed.

API base URL:

```text
http://localhost:8000
```

Interactive docs:

```text
http://localhost:8000/docs
```

OpenAPI JSON:

```text
http://localhost:8000/openapi.json
```

A static copy is also included at `openapi.json`.

## Main Endpoints

Health:

```bash
curl http://localhost:8000/
```

Pipeline:

```bash
curl -X POST http://localhost:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "input_payload": {
      "user_input": {
        "description": "Arrange a neat bedroom.",
        "room_type": "bedroom",
        "floor_area_m2": 20,
        "height": 2400,
        "shape_points": [
          {"x": 0, "y": 0},
          {"x": 2400, "y": 0},
          {"x": 2400, "y": 3500},
          {"x": 0, "y": 3500}
        ],
        "windows": 1,
        "window_direction": "SE",
        "style": "minimal"
      }
    }
  }'
```

Then poll:

```bash
curl http://localhost:8000/pipeline/{case_id}/status
curl http://localhost:8000/pipeline/{case_id}/result
```

Inventory:

```bash
curl http://localhost:8000/inventory/items
curl http://localhost:8000/inventory/types
curl "http://localhost:8000/inventory/search?q=chair"
```

Auth:

```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"password123","display_name":"Demo"}'

curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@example.com","password":"password123"}'
```

Account APIs use bearer auth:

```bash
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <access_token>"
```

## Local CLI

```bash
python run_case_backend_cli.py --input sample_input.json --user demo_user --stdout wrapped
```


Format của `GET /pipeline/{case_id}/result` là một wrapper như này:

```json
{
  "case_id": "user_xxx_20260507T083710Z",
  "result": {
    "status": "OK",
    "room": {},
    "objects": [],
    "final_style_plan": {},
    "notes": [],
    "missing": [],
    "variants": [],
    "selected_variant_id": "variant_1"
  }
}
```

Cụ thể hơn:

```json
{
  "case_id": "...",
  "result": {
    "status": "OK",
    "room": {
      "room_id": "room_1",
      "room_type": "bedroom",
      "polygon_ccw": [],
      "obstacles": [],
      "openings": {},
      "surfaces": {},
      "opening_colors": {}
    },
    "objects": [
      {
        "instance_id": "bed",
        "object_type": "bed",
        "source": "existing",
        "cluster_id": "sleep_core",
        "polygon_ccw": [
          { "x": 2300, "y": 1250 }
        ],
        "bbox": {
          "min_x": 2300,
          "min_y": 1250,
          "max_x": 4619,
          "max_y": 3556
        },
        "color_hex": "#D8CFC4",
        "material": "wood",
        "place_on": null,
        "rotation_ccw": 180,
        "front_world": { "dx": 0, "dy": -1 },
        "front_side_world": "bottom",
        "axis_world": null
      }
    ],
    "final_style_plan": {
      "style_name": "...",
      "palette": {},
      "surface_plan": {},
      "object_finish_plan": {},
      "decor_plan": {},
      "lighting_mood": "...",
      "rules": []
    },
    "notes": [],
    "missing": [],
    "variants": [
      {
        "variant_id": "variant_1",
        "label": "Option 1",
        "source": "concept:concept_04",
        "reason": "...",
        "layout_score": -4362,
        "hard_valid": false,
        "complete": true,
        "gallery_eligible": false,
        "coverage_ratio": 1.0,
        "missing_cluster_ids": [],
        "notes": [],
        "absolute_layout": {
          "status": "PARTIAL",
          "room": {},
          "openings": {},
          "objects": [],
          "placements": []
        },
        "styled_result": {
          "status": "OK",
          "room": {},
          "objects": [],
          "final_style_plan": {}
        }
      }
    ],
    "selected_variant_id": "variant_1"
  }
}
```

Điểm quan trọng: `/result` **không trả format frontend `{ position: {x,y,z}, rotation quaternion }` trực tiếp**. Nó vẫn là format nội bộ 2D theo mm:

- `styled_result.objects`: có `polygon_ccw`, `bbox`, `color_hex`, `material`.
- `absolute_layout.objects`: có `x`, `y`, `w`, `h`, `rot`, `rotation_ccw`, `bbox`, `center`.

Nếu cần output gần nhất để convert sang frontend, nên lấy:

```json
result.variants[n].absolute_layout.objects
```

hoặc variant được chọn theo:

```json
result.selected_variant_id
```

Rồi dùng layer convert/restore để đưa về tọa độ căn hộ ban đầu và quaternion cho frontend.
(API restore-output)

## Ở API normalize input
Sẽ có trường để chứa dạng đưa vào pipeline/run luôn
"system_inputs": [
  {
    "room_id": "-1.5:-2.5",
    "room_name": "Phòng ngủ 2",
    "room_type": "bedroom",
    "input_payload": { "...": "payload đưa vào hệ thống" },
    "pipeline_run_request": { "...": "copy thẳng qua /pipeline/run" }
  }
]

