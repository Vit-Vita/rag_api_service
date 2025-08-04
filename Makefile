.PHONY: setup setup_env rag_hager


# ---- For linux ------

PYTHON=python3
ENVDIR=env
#NOTEBOOK=RAG_Testing_Evaluation.ipynb


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
####### -- HAGER -- ########
############################

# Update or create the vector database
vectors_rag_hager: 
	$(ENVDIR)/bin/$(PYTHON) src/load_vectordatabase_HAGER.py

#original version
rag_hager: 
	$(ENVDIR)/bin/streamlit run src/RAG_HAGER.py

# Newest version WITHOUT dialogue or speech to text
openai_rag_hager:
	$(ENVDIR)/bin/streamlit run src/OPENAI_RAG_HAGER.py

# Newest version with speech to text and dialogue (question asked back to the candidate)
rag_dialogue_hager:
	$(ENVDIR)/bin/streamlit run src/RAG_DIALOGUE_HAGER.py

############################
####### -- LUKB-- ##########
############################

# Update or create the vector database
vectors_rag_lukb:
	$(ENVDIR)/bin/$(PYTHON) src/load_vectordatabase_LuKB.py

# Newest version WITHOUT dialogue or speech to text
openai_rag_lukb:
	$(ENVDIR)/bin/streamlit run src/OPENAI_RAG_LUKB.py

# Newest version with speech to text and dialogue (question asked back to the candidate)
rag_dialogue_lukb:
	$(ENVDIR)/bin/streamlit run src/RAG_DIALOGUE_LUKB.py


############################
####### -- RICOLA-- ##########
############################

rag_dialogue_ricola:
	$(ENVDIR)/bin/streamlit run src/RAG_DIALOGUE_RICOLA.py

diagnose_page:
	$(ENVDIR)/bin/python src/diagnose_page.py


#################################################################################################
#################################################################################################

# --- For windows ---

#PYTHON=python
#ENVDIR=env
#NOTEBOOK=RAG_Testing_Evaluation.ipynb

############################
### -- ENVIRONMENT -- ######
############################

#setup_env:
#	python -m venv $(ENVDIR)

#setup: setup_env
#	$(ENVDIR)/Scripts/pip install -r requirements.txt

#clean:
#	rmdir env /s /q

############################
####### -- HAGER -- ########
############################

#rag_hager: 
#	$(ENVDIR)/Scripts/streamlit run src/RAG_HAGER.py

#openai_rag_hager:
#	$(ENVDIR)/Scripts/streamlit run src/BGE_RAG_HAGER.py

#vectors_rag_hager: 
#	$(ENVDIR)/Scripts/$(PYTHON) src/load_vectordatabase_HAGER.py

#rag_dialogue_hager:
#	$(ENVDIR)/Scripts/streamlit run src/RAG_DIALOGUE_HAGER.py

############################
####### -- LUKB-- ##########
############################

# Update or create the vector database
#vectors_rag_lukb:
#	$(ENVDIR)/Scripts/$(PYTHON) src/load_vectordatabase_LuKB.py

# Newest version WITHOUT dialogue or speech to text
#openai_rag_lukb:
#	$(ENVDIR)/Scripts/streamlit run src/OPENAI_RAG_LUKB.py

# Newest version with speech to text and dialogue (question asked back to the candidate)
#rag_dialogue_lukb:
#	$(ENVDIR)/Scripts/streamlit run src/RAG_DIALOGUE_LUKB.py
