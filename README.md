
# Precision Mapping:
A python based individualized fMRI brain parcellation package with block matrix gpu acceleration

# Installation and Usage
Download git repo

```
git clone git@github.com:cole-khamnei/precision_mapping.git
```

Example Usage:
```
python precision_mapping/precision_mapping.py -c cifti1.dtseries.nii -s "SUBJECT01" -l "SUBJECT01_RUN1" -o output_dir/
```

Can also be imported as a package:

```
import precision_mapping as pm

ps = pm.precision_mapping(dtseries_path, subject_id, sample_label, out_dir, overwrite=True)
```

![example_image](tests/outputs/example_parcellation_plot.png)


# TODO:
## deployable package todos:
	[]  add frame sensoring files
		[] working, just need to find example '.dat' file or w/e the fileext is
	
	[] add smoothing distance filter
		[] do you exclude connections that are within the kernel size?
			[] if yes, then can just pass the filter matrix I think


[] benchmark on HCP 7T
	[] run first day second day
	[] repeat on 3T and determine "ability to detect" / "power"

[] infomaps stability benchmarking with parcels:
	[] Add in noise?a



[] stability benchmarks:
	[] function to calcualte R2 between 2 vertex level FC maps
		[] Can also do it at the vertex level, thus leading to "vertex FC similarity maps" between subjects
			[] can do for SzP
			[] can do for twins
