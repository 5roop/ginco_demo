# ginco_demo

Using GInCo to train a simpletransformers model

In `demo.ipynb` we demonstrate how to load the GInCo dataset, train a `simpletransformers` model and evaluate it on test data. 

*Work in progress.* 

Ideas for improvement:
* Once the dataset is published: do we download it first and then use load it? Alternative: use pandas to read it directly from web.
* Add citation information, link to the paper, ...
* After proper dataset publication possibly prepare a HugginFace dataset.

# Dataset description - GINCO - Genre IdeNtification COrpus

## Files

This dataset consists of two files: `suitable.json` and `nonsuitable.json`.

## File structure

Each file is a json packaged list of documents. Documents in the `suitable.json` file have the following fields:

| field             | description                                                    | remarks                                                                                                            |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `id`                | string, document unique id                                      |                                                                                                                    |
| `url`               | string, url of the original website from which the text was scrapped   |                                                                                                                    |
| `crawled`           | string, crawl year                                             |                                                                                                                    |
| `hard`              | boolean, human annotated, whether the document was hard to label        |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata |                                                                                                                    |
| `primary_level_N`   | string, human annotated primary label                                  | `N` ∈ {1,2,3} indicates label downsampling extent, 1 being not at all and 3 being downcast from 24 labels to 12    |
| `secondary_level_N` | string, human annotated secondary label                                | see above                                                                                                             |
| `tertiary_level_N`  | string, human annotated primary label                                  | see above                                                                                                             |
| `split`             | string, whether the document belongs to *train*, *dev*, or *test* split      | 60:20:20 split, stratified by primary_level_2, with documents with the same domain strictly kept in the same split |
| `domain`            | string, domain from which the document was scrapped                    | parsed from `url` field                                                                                              |



Documents in the `nonsuitable.json` file have the following fields:

| field             | description                                                    | remarks                                                                                                            |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `id`                | string, document unique string id                                      |                                                                                                                    |
| `url`               | string, url of the original website from which the text was scrapped   |                                                                                                                    |
| `crawled`           | string, crawl year                                             |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata |                                                                                                                    |
| `primary_level_1`   | string, human annotated primary label                                  | there was no downcasting for unsuitable dataset|
| `secondary_level_1` | string, human annotated secondary label                                |                                                                                                                    |
| `split`             | string, whether the document belongs to train, dev, or test split      | 60:20:20 split, stratified by primary_level_1|
| `domain`            | string, domain from which the document was scrapped                    | parsed from `url` field                                                                                              |


## Paragraph structure

Items of the list in `paragraphs` have the following fields:

| field     | description                                                                | remarks                             |
|-----------|----------------------------------------------------------------------------|-------------------------------------|
| `text`      | string, paragraph text                                                   |                                     |
| `duplicate` | boolean, result of automated deduplication                               |                                     |
| `keep`      | boolean, human annotated tag whether or not the paragraph should be kept | only present in suitable paragraphs |


