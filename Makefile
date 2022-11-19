env-create:
	tox -e zero_deforestation

env-compile:
	pip-compile requirements.in

test:
	pytest tests

lint:
	pylint --rcfile zero_deforestation/.pylintrc zero_deforestation

download-data:
	cd zero_deforestation/data && \
	wget -q https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/train.csv && \
	wget -q https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/test.csv && \
	wget -q https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/SchneiderElectricEuropeanHackathon22/train_test_data.zip && \
	unzip -qq train_test_data.zip && \
	wget -q http://download.cs.stanford.edu/deep/ForestNetDataset.zip && \
	unzip -qq ForestNetDataset.zip && \
	echo Finished!

create-extended-dataset:
	PYTHONPATH=. python zero_deforestation/scripts/create_extended_dataset.py

split-dataset:
	PYTHONPATH=. python zero_deforestation/scripts/split_train_val.py $(DATASET) $(TRAIN_PROPORTION)