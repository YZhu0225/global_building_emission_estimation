# Run the project
run:
	python3 run.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f data/temp/*

clean-processed:
	rm -f data/processed/*

# clean-all: clean clean-processed

# NOTE FOR WINDOWS USERS: Make is more challenging to install on Windows.
# Recommend using Chocolately guide here: 
# https://earthly.dev/blog/makefiles-on-windows/