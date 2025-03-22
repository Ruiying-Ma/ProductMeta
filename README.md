# ProductMeta

## Preliminaries
### Add your API key
Create a .env file under the root directory:

```.env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_MODEL_GPT=
AZURE_OPENAI_MODEL_DALLE=
```

### Install python packages

## Core Code
Check [ProductMetaSourceGen.py](./ProductMetaSourceGen.py).

## Guidance
1. Execute

    Run 
    ```bash
    python ProductMetaSourceGen.py
    ```

    The result will be stored in `log/` folder.
    > Example: [kettle-example-log/](./kettle-example-log/)

2. Visualize

    Run 
    ```bash
    python Visualizer.py
    ```

    The result will be stored in `log/figure/` folder.
    > Example: [kettle-example-log/figure/](./kettle-example-log/figure/)