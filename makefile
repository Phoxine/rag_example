install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
.PHONY: install

index:
	. .venv/bin/activate && python 1_indexing.py
.PHONY: index

agent:
	. .venv/bin/activate && python 2_rag_agent.py
.PHONY: agent

chain:
	. .venv/bin/activate && python 3_rag_chain.py
.PHONY: chain

rm_db:
	rm -rf chroma_db
.PHONY: rm_db
