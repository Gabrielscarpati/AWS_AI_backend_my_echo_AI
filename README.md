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
/Users/gbscarpati/Desktop/development/AWS_AI_backend_my_echo_AI/venv/bin/python /Users/gbscarpati/Desktop/development/AWS_AI_backend_my_echo_AI/local_test.py



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

export AWS_REGION=us-east-1
export ACCOUNT_ID=827138162380
export REPO_NAME=influencer-brain-api
export IMAGE_TAG=latest && \

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 827138162380.dkr.ecr.us-east-1.amazonaws.com && \

docker buildx build --platform linux/amd64 -t influencer-brain-api:latest /Users/gbscarpati/Desktop/development/AWS_AI_backend_my_echo_AI/image && \

docker tag influencer-brain-api:latest 827138162380.dkr.ecr.us-east-1.amazonaws.com/influencer-brain-api:latest && \

docker push 827138162380.dkr.ecr.us-east-1.amazonaws.com/influencer-brain-api:latest && \

aws ecs update-service --cluster influencer-brain --service influencer-brain-task-service-wepvg641 --force-new-deployment --region us-east-1