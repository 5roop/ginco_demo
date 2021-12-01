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

| field               | description                                                             | remarks                                                                                                            |
| ------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `id`                | string, document unique id                                              |                                                                                                                    |
| `url`               | string, url of the original website from which the text was scrapped    |                                                                                                                    |
| `crawled`           | string, crawl year                                                      |                                                                                                                    |
| `hard`              | boolean, human annotated, whether the document was hard to label        |                                                                                                                    |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata          |                                                                                                                    |
| `primary_level_1`   | string, human annotated primary label                                   | original primary label                                                                                             |
| `secondary_level_1` | string, human annotated secondary label                                 | original secondary label                                                                                           |
| `tertiary_level_1`  | string, human annotated tertiary label                                  | original tertiary label                                                                                            |
| `primary_level_2`   | string, primary label after first downcasting                           | 21 labels in label set, basis for stratified train:dev:test split                                                  |
| `secondary_level_2` | string, secondary label after first downcasting                         | 21 labels in label set                                                                                             |
| `tertiary_level_2`  | string, tertiary label after first downcasting                          | 21 labels in label set                                                                                             |
| `primary_level_3`   | string, primary label after second downcasting                          | 12 labels in label set                                                                                             |
| `secondary_level_3` | string, secondary label after second downcasting                        | 12 labels in label set                                                                                             |
| `tertiary_level_3`  | string, tertiary label after second downcasting                         | 12 labels in label set                                                                                             |
| `split`             | string, whether the document belongs to *train*, *dev*, or *test* split | 60:20:20 split, stratified by primary_level_2, with documents with the same domain strictly kept in the same split |
| `domain`            | string, domain from which the document was scrapped                     | parsed from `url` field                                                                                            |



Documents in the `nonsuitable.json` file have the following fields:

| field               | description                                                          | remarks                                         |
| ------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| `id`                | string, document unique string id                                    |                                                 |
| `url`               | string, url of the original website from which the text was scrapped |                                                 |
| `crawled`           | string, crawl year                                                   |                                                 |
| `paragraphs`        | list of dictionaries, containing paragraphs and their metadata       |                                                 |
| `primary_level_1`   | string, human annotated primary label                                | there was no downcasting for unsuitable dataset |
| `secondary_level_1` | string, human annotated secondary label                              |                                                 |
| `split`             | string, whether the document belongs to train, dev, or test split    | 60:20:20 split, stratified by primary_level_1   |
| `domain`            | string, domain from which the document was scrapped                  | parsed from `url` field                         |


## Paragraph structure

Items of the list in `paragraphs` have the following fields:

| field       | description                                                              | remarks                             |
| ----------- | ------------------------------------------------------------------------ | ----------------------------------- |
| `text`      | string, paragraph text                                                   |                                     |
| `duplicate` | boolean, result of automated deduplication                               |                                     |
| `keep`      | boolean, human annotated tag whether or not the paragraph should be kept | only present in suitable paragraphs |



## Sample data instance

```
{'id': '3949',
 'url': 'http://www.pomurje.si/aktualno/sport/zimska-liga-malega-nogometa/',
 'crawled': '2014',
 'hard': False,
 'paragraphs': [{'text': 'Šport', 'duplicate': False, 'keep': True},
  {'text': 'Zimska liga malega nogometa sobota, 12.02.2011',
   'duplicate': False,
   'keep': True},
  {'text': 'avtor: Tonček Gider', 'duplicate': False, 'keep': True},
  {'text': "V 7. krogu zimske lige v malem nogometu v Križevcih pri Ljutomeru je v prvi ligi vodilni 100 plus iz Križevec izgubil s tretjo ekipo na lestvici Rock'n roll iz Križevec z rezultatom 1:2, druga na lestvici Top Finedika iz Križevec je bila poražena z ekipo Bar Milene iz Ključarovec z rezultatom 7:8. V drugi križevski ligi je vodilni Cafe del Mar iz Vučje vasi premagal Montažo Vrbnjak iz Stare Nove vasi z rezultatom 3:2.",
   'duplicate': False,
   'keep': True},
  {'text': 'oglasno sporočilo', 'duplicate': False, 'keep': True},
  {'text': 'Ocena', 'duplicate': False, 'keep': True},
  {'text': 'Komentiraj Za komentiranje ali ocenjevanje moraš biti registriran in prijavljen uporabnik. Registriraj se!',
   'duplicate': True,
   'keep': False}],
 'primary_level_1': 'News/Reporting',
 'primary_level_2': 'News/Reporting',
 'primary_level_3': 'News/Reporting',
 'secondary_level_1': '',
 'secondary_level_2': '',
 'secondary_level_3': '',
 'tertiary_level_1': '',
 'tertiary_level_2': '',
 'tertiary_level_3': '',
 'split': 'test',
 'domain': 'www.pomurje.si',
}
 ```

 ## Downsampling

 `primary_level_1`, `secondary_level_1`, and `tertiary_level_1` labels consist of the original labels (24 labels). Some categories had less than 5 instances and they were relabeled into `Other`, and we record this labeling in fields  `primary_level_2`, `secondary_level_2`, and `tertiary_level_2`. Another downsampling was used to further reduce the label set to 12 labels, saved in fields `primary_level_3`, `secondary_level_3`, and `tertiary_level_3`.

 For the first downcasting (level 1 to level 2) we used the following mapping:
 ```
 {"Script/Drama":"Other", "Lyrical":"Other","FAQ":"Other"}
 ```

 And the second downcasting (level 2 to level 3) uses the mapping:
 ```
 {"Recipe":"Instruction", "Research Article":"Information/Explanation", "Review":"Opinion/Argumentation", "Promotion of Services":"Promotion", "Promotion of a Product":"Promotion", "Invitation":"Promotion", "Correspondence":"Other", "Prose":"Other", "Call":"Other"}
 ```

