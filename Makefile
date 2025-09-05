.PHONY: setup setup_env rag_hager


# ---- For linux ------

PYTHON=python3
ENVDIR=env


############################
### -- ENVIRONMENT -- ######
############################
setup_env:
	$(PYTHON) -m venv $(ENVDIR)

setup: setup_env
	$(ENVDIR)/bin/pip install -r requirements.txt

register_kernel:
	source $(ENVDIR)/bin/activate && \
	pip install ipykernel notebook && \
	python -m ipykernel install --user --name=env --display-name "Python (env)"
	
clean:
	rm -rf $(ENVDIR)

############################
####### -- API -- ########
############################

# Update or create the vector database
vectors_rag_api_all: 
	$(ENVDIR)/bin/$(PYTHON) load_relational_and_vector_databases.py

