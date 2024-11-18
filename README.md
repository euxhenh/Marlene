## Recovering Time-Varying Networks From Single-Cell Data

This is the code for Marlene, presented in the paper https://arxiv.org/abs/2410.01853.

Config files used for training can be found under `configs`

Download the data from following instructions in the respective papers
https://www.nature.com/articles/s41591-023-02327-2,
https://www.nature.com/articles/s41590-023-01608-9 and https://www.nature.com/articles/s41467-020-17358-3. Convert the expression
data to AnnData using the `anndata` package and place the files under the
`data` folder.

First install Marlene from `setup.py`
```bash
pip install -e .
```

To run Marlene, use
```bash
python train.py --conf configs/pmbc.ini --runid your-run-id
```
or
```bash
python train.py --conf configs/hlca.ini --runid your-run-id
```

The main model class for Marlene is under `marlene/models/marlene.py`.
