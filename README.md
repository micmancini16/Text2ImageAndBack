## Text2Image

Text2Image is a library that allows to perform text 2 image semantic search: provide a text description, ex. "I want a long sleeved green dress", the algorithm will look for long sleeved green dress images and return them back.

It can also be used to perform correlated tasks such as image2image search, image2text search or zero shot classification.

## Installation 

Pull the repository, create a virtual environment and install requirements

            pip install -r requirements.txt

## Demo scripts

Run `python demo_create_index.py` to create an index for the `ceyda/fashion-products-small` dataset, a publicly available fashion dataset. Indexing might require an hour with the current library implementation. Indexing produce a pickle file that will be recalled by other scripts. This is usually a one-time job.

Run `python demo_text2image.py` to see how the library can be used to run queries against the previously computed index.

Run `python demo_zero_shot_classification.py` to see an example of how the library can be also used to run zero shot classification. The script runs zero shot classification against the fashion dataset, achieving 76% accuracy.

The notebooks contains further examples of how the library can be used.