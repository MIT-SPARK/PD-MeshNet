## Unit tests
To run the unit tests, simply navigate to the `test/` folder of this repo and run the following command within the `virtualenv` created upon installation:
```bash
python -m unittest
```
The command above will run all the available unit tests.

Alternatively, one can:
- Run all the unit tests for a certain component (e.g., datasets), by navigating to the corresponding subfolder of the `test/` folder (e.g., `test/datasets/`) and running the same command as above;
- Run a single unit-test script for a certain component. For instance, to run the script [`test_shrec2016_dual_primal.py`](../test/datasets/test_shrec2016_dual_primal.py) from the dataset unit tests, navigate to `test/datasets/` and run:
    ```bash
    python -m unittest test_shrec2016_dual_primal.py
    ```