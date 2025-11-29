TODO:

# deployable package todos:
	[]  add frame sensoring files
		[] just second arg for anything that takes in a dtseries	
			[] check the sensoring file (some asserts if needed), load in as np array 0,1s
			[] pass the sensor file to the load dtseries function (and sensor there)
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

