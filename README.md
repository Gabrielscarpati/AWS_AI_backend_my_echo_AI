# Welcome to your CDK TypeScript project

This is a blank project for CDK development with TypeScript.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Useful commands

* `npm run build`   compile typescript to js
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `npx cdk deploy`  deploy this stack to your default AWS account/region
* `npx cdk diff`    compare deployed stack with current state
* `npx cdk synth`   emits the synthesized CloudFormation template



ADD data to pinecone.
python3 - <<'PY'
import runpy, uuid, datetime
mod = runpy.run_path("image/src/influencer_data_entry.py")
load_from_json = mod['load_from_json']
upload_records = mod['upload_records']

data = load_from_json("image/src/reflections.json")
creator = data.get("creator_id")

records = []
for r in data.get("reflections", []):
    r.setdefault("id", f"ref-{uuid.uuid4().hex}")
    r["type"] = "reflection"
    r["creator_id"] = creator
    r.setdefault("created_at", datetime.datetime.utcnow().isoformat() + "Z")
    records.append(r)

for m in data.get("memories", []):
    m.setdefault("id", f"mem-{uuid.uuid4().hex}")
    m["type"] = "memory"
    m["creator_id"] = creator
    m.setdefault("created_at", datetime.datetime.utcnow().isoformat() + "Z")
    records.append(m)

upload_records(records)
PY