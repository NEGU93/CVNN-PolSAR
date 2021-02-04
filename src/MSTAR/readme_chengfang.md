# Instructions to run simulations

## 1. Read data

In this step you will convert the [MSTAR dataset](https://www.sdms.afrl.af.mil/index.php?collection=mstar)

You must download and untar:

1. MSTAR / IU Mixed Targets
    - CD 1 (341 MB)
    - CD 2 (336 MB)
2. MSTAR Target Chips (T72 BMP2 BTR70 SLICY)
    - CD 1 (208 MB)

Now changed the `path` variable at `mstar_raw_data_reader.py` with the folder where you untared the 3 cd's and launch.

## 2. Monte carlo simulations

1. Change the `path` variable at `mstar_data_processing.py` with the folder where you untared the 3 cd's 
2. Create your own `CVNN` model (compiled with loss and optimizer)
3. Run `do_monte_carlo` giving as a parameter the complex model.

Note: You can change the parameters with the `montecarlo_config` dictionary variable.