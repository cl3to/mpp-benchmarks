#include "XSbench_header.h"

////////////////////////////////////////////////////////////////////////////////////
// BASELINE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////
// All "baseline" code is at the top of this file. The baseline code is a simple
// implementation of the algorithm, with only minor CPU optimizations in place.
// Following these functions are a number of optimized variants,
// which each deploy a different combination of optimizations strategies. By
// default, XSBench will only run the baseline implementation. Optimized variants
// are not yet implemented for this OpenMP targeting offload port.
////////////////////////////////////////////////////////////////////////////////////

unsigned long long run_event_based_simulation(Inputs in, SimulationData SD, int mype)
{
	if( mype == 0)	
		printf("Beginning event based simulation...\n");
	
	////////////////////////////////////////////////////////////////////////////////
	// SUMMARY: Simulation Data Structure Manifest for "SD" Object
	// Here we list all heap arrays (and lengths) in SD that would need to be
	// offloaded manually if using an accelerator with a seperate memory space
	////////////////////////////////////////////////////////////////////////////////
	// int * num_nucs;                     // Length = length_num_nucs;
	// double * concs;                     // Length = length_concs
	// int * mats;                         // Length = length_mats
	// double * unionized_energy_array;    // Length = length_unionized_energy_array
	// int * index_grid;                   // Length = length_index_grid
	// NuclideGridPoint * nuclide_grid;    // Length = length_nuclide_grid
	// 
	// Note: "unionized_energy_array" and "index_grid" can be of zero length
	//        depending on lookup method.
	//
	// Note: "Lengths" are given as the number of objects in the array, not the
	//       number of bytes.
	////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////////////////////////
	// Begin Actual Simulation Loop 
	////////////////////////////////////////////////////////////////////////////////
	int num_devices = omp_get_num_devices();
	unsigned long chunk = in.lookups;
    
	int *mats_d[num_devices];
	int *num_nucs_d[num_devices];
	int *index_grid_d[num_devices];
	double *concs_d[num_devices];
	double *unionized_energy_arr_d[num_devices];
	NuclideGridPoint *nuclide_grid_d[num_devices];

	int max_num_nucs = SD.max_num_nucs;
	int host_device = omp_get_initial_device();

	int deps[6*num_devices] = {};

	printf("Num Devices: %d\nChunk Size: %lu\n", num_devices, chunk);
 
	#pragma omp parallel for num_threads(num_devices)
	for (int K = 0; K < num_devices; K++) {
		num_nucs_d[K] = (int *) omp_target_alloc(SD.length_num_nucs*sizeof(int), K);
		concs_d[K] = (double *) omp_target_alloc(SD.length_concs*sizeof(double), K);
		mats_d[K] = (int *) omp_target_alloc(SD.length_mats*sizeof(int), K);
		unionized_energy_arr_d[K] = (double *) omp_target_alloc(SD.length_unionized_energy_array*sizeof(double), K);
		index_grid_d[K] = (int *) omp_target_alloc(SD.length_index_grid*sizeof(int), K);
		nuclide_grid_d[K] = (NuclideGridPoint *) omp_target_alloc(SD.length_nuclide_grid*sizeof(NuclideGridPoint), K);
	}

	#pragma omp parallel num_threads(num_devices)
	#pragma omp single
	for (int K = 0; K < num_devices; K++) {

		int *num_nucs_dk = num_nucs_d[K];
		double *concs_dk = concs_d[K];
		int *mats_dk = mats_d[K];
		double *unionized_energy_arr_dk = unionized_energy_arr_d[K];
		int *index_grid_dk = index_grid_d[K];
		NuclideGridPoint *nuclide_grid_dk = nuclide_grid_d[K];

		if (K == 0 || K == 4) {

			#pragma omp task depend(out: deps[6*K])
			{
				// printf("(K=%d): source_device=%d. dep=%p\n", K, host_device, &deps[6*K]);
				omp_target_memcpy(num_nucs_dk, SD.num_nucs, SD.length_num_nucs*sizeof(int), 0 , 0, K, host_device);			
			}

			#pragma omp task depend(out: deps[6*K + 1])
			{
				// printf("if(K=%d): dep=%p\n", K, &deps[6*K + 1]);
				omp_target_memcpy(concs_dk, SD.concs, SD.length_concs*sizeof(double), 0 , 0, K, host_device);
			}

			#pragma omp task depend(out: deps[6*K + 2])
			{
				// printf("if(K=%d): dep=%p\n", K, &deps[6*K + 2]);
				omp_target_memcpy(mats_dk, SD.mats, SD.length_mats*sizeof(int), 0 , 0, K, host_device);
			}

			#pragma omp task depend(out: deps[6*K + 3])
			{
				// printf("if(K=%d): dep=%p\n", K, &deps[6*K + 3]);
				omp_target_memcpy(unionized_energy_arr_dk, SD.unionized_energy_array, SD.length_unionized_energy_array*sizeof(double), 0 , 0, K, host_device);
			}

			#pragma omp task depend(out: deps[6*K + 4])
			{
				// printf("if(K=%d): dep=%p\n", K, &deps[6*K + 4]);
				omp_target_memcpy(index_grid_dk, SD.index_grid, SD.length_index_grid*sizeof(int), 0 , 0, K, host_device);
			}

			#pragma omp task depend(out: deps[6*K + 5])
			{
				// printf("if(K=%d): dep=%p\n", K, &deps[6*K + 5]);
				omp_target_memcpy(nuclide_grid_dk, SD.nuclide_grid, SD.length_nuclide_grid*sizeof(NuclideGridPoint), 0 , 0, K, host_device);
			}
		}

		if (K % 4 == 0) {
			int source_device = K;
			int left_device = 2*K+4;
			int right_device = 2*K+8;

			if (left_device < num_devices && K != 0) {
				#pragma omp task depend(in: deps[6*source_device]) depend(out: deps[6*left_device])
				{
					// printf("(K=%d): source_device=%d, dep=%p, num_nucs_dk=%p\n", left_device, source_device, &deps[6*source_device], num_nucs_dk);
					omp_target_memcpy(num_nucs_d[left_device], num_nucs_d[source_device], SD.length_num_nucs*sizeof(int), 0 , 0, left_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 1]) depend(out: deps[6*left_device + 1])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, concs_dk=%p\n", K, source_device, &deps[6*source_device + 1], concs_dk);
					omp_target_memcpy(concs_d[left_device], concs_d[source_device], SD.length_concs*sizeof(double), 0 , 0, left_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 2]) depend(out: deps[6*left_device + 2])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, mats_dk=%p\n", K, source_device, &deps[6*source_device + 2], mats_dk);
					omp_target_memcpy(mats_d[left_device], mats_d[source_device], SD.length_mats*sizeof(int), 0 , 0, left_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 3]) depend(out: deps[6*left_device + 3])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, unionized_energy_arr_dk=%p\n", K, source_device, &deps[6*source_device + 3], unionized_energy_arr_dk);
					omp_target_memcpy(unionized_energy_arr_d[left_device], unionized_energy_arr_d[source_device], SD.length_unionized_energy_array*sizeof(double), 0 , 0, left_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 4]) depend(out: deps[6*left_device + 4])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, index_grid_dk=%p\n", K, source_device, &deps[6*source_device + 4], index_grid_dk);
					omp_target_memcpy(index_grid_d[left_device], index_grid_d[source_device], SD.length_index_grid*sizeof(int), 0 , 0, left_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 5]) depend(out: deps[6*left_device + 5])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, nuclide_grid_dk=%p\n", K, source_device, &deps[6*source_device + 5], nuclide_grid_dk);
					omp_target_memcpy(nuclide_grid_d[left_device], nuclide_grid_d[source_device], SD.length_nuclide_grid*sizeof(NuclideGridPoint), 0 , 0, left_device, source_device);						
				}				
			}
	
			if (right_device < num_devices) {
				#pragma omp task depend(in: deps[6*source_device]) depend(out: deps[6*right_device])
				{
					// printf("(K=%d): source_device=%d, dep=%p, num_nucs_dk=%p\n", right_device, source_device, &deps[6*source_device], num_nucs_dk);
					omp_target_memcpy(num_nucs_d[right_device], num_nucs_d[source_device], SD.length_num_nucs*sizeof(int), 0 , 0, right_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 1]) depend(out: deps[6*right_device + 1])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, concs_dk=%p\n", K, source_device, &deps[6*source_device + 1], concs_dk);
					omp_target_memcpy(concs_d[right_device], concs_d[source_device], SD.length_concs*sizeof(double), 0 , 0, right_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 2]) depend(out: deps[6*right_device + 2])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, mats_dk=%p\n", K, source_device, &deps[6*source_device + 2], mats_dk);
					omp_target_memcpy(mats_d[right_device], mats_d[source_device], SD.length_mats*sizeof(int), 0 , 0, right_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 3]) depend(out: deps[6*right_device + 3])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, unionized_energy_arr_dk=%p\n", K, source_device, &deps[6*source_device + 3], unionized_energy_arr_dk);
					omp_target_memcpy(unionized_energy_arr_d[right_device], unionized_energy_arr_d[source_device], SD.length_unionized_energy_array*sizeof(double), 0 , 0, right_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 4]) depend(out: deps[6*right_device + 4])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, index_grid_dk=%p\n", K, source_device, &deps[6*source_device + 4], index_grid_dk);
					omp_target_memcpy(index_grid_d[right_device], index_grid_d[source_device], SD.length_index_grid*sizeof(int), 0 , 0, right_device, source_device);
				}
	
				#pragma omp task depend(in: deps[6*source_device + 5]) depend(out: deps[6*right_device + 5])
				{
					// printf("else(K=%d): source_device=%d, dep=%p, nuclide_grid_dk=%p\n", K, source_device, &deps[6*source_device + 5], nuclide_grid_dk);
					omp_target_memcpy(nuclide_grid_d[right_device], nuclide_grid_d[source_device], SD.length_nuclide_grid*sizeof(NuclideGridPoint), 0 , 0, right_device, source_device);						
				}				
			}
		}

		else {
			int source_device = (K/4) * 4;

			#pragma omp task depend(in: deps[6*source_device])
			{
				// printf("(K=%d): source_device=%d, dep=%p, num_nucs_dk=%p\n", K, source_device, &deps[6*source_device], num_nucs_dk);
				omp_target_memcpy(num_nucs_dk, num_nucs_d[source_device], SD.length_num_nucs*sizeof(int), 0 , 0, K, source_device);
			}

			#pragma omp task depend(in: deps[6*source_device + 1])
			{
				// printf("else(K=%d): source_device=%d, dep=%p, concs_dk=%p\n", K, source_device, &deps[6*source_device + 1], concs_dk);
				omp_target_memcpy(concs_dk, concs_d[source_device], SD.length_concs*sizeof(double), 0 , 0, K, source_device);
			}

			#pragma omp task depend(in: deps[6*source_device + 2])
			{
				// printf("else(K=%d): source_device=%d, dep=%p, mats_dk=%p\n", K, source_device, &deps[6*source_device + 2], mats_dk);
				omp_target_memcpy(mats_dk, mats_d[source_device], SD.length_mats*sizeof(int), 0 , 0, K, source_device);
			}

			#pragma omp task depend(in: deps[6*source_device + 3])
			{
				// printf("else(K=%d): source_device=%d, dep=%p, unionized_energy_arr_dk=%p\n", K, source_device, &deps[6*source_device + 3], unionized_energy_arr_dk);
				omp_target_memcpy(unionized_energy_arr_dk, unionized_energy_arr_d[source_device], SD.length_unionized_energy_array*sizeof(double), 0 , 0, K, source_device);
			}

			#pragma omp task depend(in: deps[6*source_device + 4])
			{
				// printf("else(K=%d): source_device=%d, dep=%p, index_grid_dk=%p\n", K, source_device, &deps[6*source_device + 4], index_grid_dk);
				omp_target_memcpy(index_grid_dk, index_grid_d[source_device], SD.length_index_grid*sizeof(int), 0 , 0, K, source_device);
			}

			#pragma omp task depend(in: deps[6*source_device + 5])
			{
				// printf("else(K=%d): source_device=%d, dep=%p, nuclide_grid_dk=%p\n", K, source_device, &deps[6*source_device + 5], nuclide_grid_dk);
				omp_target_memcpy(nuclide_grid_dk, nuclide_grid_d[source_device], SD.length_nuclide_grid*sizeof(NuclideGridPoint), 0 , 0, K, source_device);						
			}
		}
		
	}

	#pragma omp parallel for num_threads(num_devices)
	for (int K = 0; K < num_devices; K++) {

		int *num_nucs_dk = num_nucs_d[K];
		double *concs_dk = concs_d[K];
		int *mats_dk = mats_d[K];
		double *unionized_energy_arr_dk = unionized_energy_arr_d[K];
		int *index_grid_dk = index_grid_d[K];
		NuclideGridPoint *nuclide_grid_dk = nuclide_grid_d[K];

		// #pragma omp task depend(in: deps[6*K], deps[6*K+1], deps[6*K+2], deps[6*K+3], deps[6*K+4], deps[6*K+5])
		#pragma omp target teams distribute parallel for \
		        is_device_ptr(num_nucs_dk, concs_dk, mats_dk, unionized_energy_arr_dk, index_grid_dk, nuclide_grid_dk) \
				firstprivate(max_num_nucs) \
				device(K)
		for(unsigned long i = 0; i < chunk; i++)
		{
			// Set the initial seed value
			uint64_t seed = STARTING_SEED;	

			// Forward seed to lookup index (we need 2 samples per lookup)
			seed = fast_forward_LCG(seed, 2*i);

			// Randomly pick an energy and material for the particle
			double p_energy = LCG_random_double(&seed);
			int mat         = pick_mat(&seed); 

			// debugging
			//printf("E = %lf mat = %d\n", p_energy, mat);

			double macro_xs_vector[5] = {0};
			
			// Perform macroscopic Cross Section Lookup
			calculate_macro_xs(
				p_energy,        // Sampled neutron energy (in lethargy)
				mat,             // Sampled material type index neutron is in
				in.n_isotopes,   // Total number of isotopes in simulation
				in.n_gridpoints, // Number of gridpoints per isotope in simulation
				num_nucs_dk,     // 1-D array with number of nuclides per material
				concs_dk,        // Flattened 2-D array with concentration of each nuclide in each material
				unionized_energy_arr_dk, // 1-D Unionized energy array
				index_grid_dk,   // Flattened 2-D grid holding indices into nuclide grid for each unionized energy level
				nuclide_grid_dk, // Flattened 2-D grid holding energy levels and XS_data for all nuclides in simulation
				mats_dk,         // Flattened 2-D array with nuclide indices defining composition of each type of material
				macro_xs_vector, // 1-D array with result of the macroscopic cross section (5 different reaction channels)
				in.grid_type,    // Lookup type (nuclide, hash, or unionized)
				in.hash_bins,    // Number of hash bins used (if using hash lookup type)
				max_num_nucs     // Maximum number of nuclides present in any material
			);

			// For verification, and to prevent the compiler from optimizing
			// all work out, we interrogate the returned macro_xs_vector array
			// to find its maximum value index, then increment the verification
			// value by that index. In this implementation, we prevent thread
			// contention by using an OMP reduction on the verification value.
			// For accelerators, a different approach might be required
			// (e.g., atomics, reduction of thread-specific values in large
			// array via CUDA thrust, etc).
			double max = -1.0;
			int max_idx = 0;
			for(int j = 0; j < 5; j++ )
			{
				if( macro_xs_vector[j] > max )
				{
					max = macro_xs_vector[j];
					max_idx = j;
				}
			}
		}

		omp_target_free(num_nucs_d[K], K);
		omp_target_free(concs_d[K], K);
		omp_target_free(mats_d[K], K);
		omp_target_free(unionized_energy_arr_d[K], K);
		omp_target_free(index_grid_d[K], K);
		omp_target_free(nuclide_grid_d[K], K);
	}

	return 0;
}

// Calculates the microscopic cross section for a given nuclide & energy
void calculate_micro_xs(   double p_energy, int nuc, long n_isotopes,
                           long n_gridpoints,
                           double *  egrid, int *  index_data,
                           NuclideGridPoint *  nuclide_grids,
                           long idx, double *  xs_vector, int grid_type, int hash_bins ){
	// Variables
	double f;
	NuclideGridPoint * low, * high;

	// If using only the nuclide grid, we must perform a binary search
	// to find the energy location in this particular nuclide's grid.
	if( grid_type == NUCLIDE )
	{
		// Perform binary search on the Nuclide Grid to find the index
		idx = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], 0, n_gridpoints-1);

		// pull ptr from nuclide grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( idx == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + idx - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + idx];
	}
	else if( grid_type == UNIONIZED) // Unionized Energy Grid - we already know the index, no binary search needed.
	{
		// pull ptr from energy grid and check to ensure that
		// we're not reading off the end of the nuclide's grid
		if( index_data[idx * n_isotopes + nuc] == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc] - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + index_data[idx * n_isotopes + nuc]];
	}
	else // Hash grid
	{
		// load lower bounding index
		int u_low = index_data[idx * n_isotopes + nuc];

		// Determine higher bounding index
		int u_high;
		if( idx == hash_bins - 1 )
			u_high = n_gridpoints - 1;
		else
			u_high = index_data[(idx+1)*n_isotopes + nuc] + 1;

		// Check edge cases to make sure energy is actually between these
		// Then, if things look good, search for gridpoint in the nuclide grid
		// within the lower and higher limits we've calculated.
		double e_low  = nuclide_grids[nuc*n_gridpoints + u_low].energy;
		double e_high = nuclide_grids[nuc*n_gridpoints + u_high].energy;
		int lower;
		if( p_energy <= e_low )
			lower = 0;
		else if( p_energy >= e_high )
			lower = n_gridpoints - 1;
		else
			lower = grid_search_nuclide( n_gridpoints, p_energy, &nuclide_grids[nuc*n_gridpoints], u_low, u_high);

		if( lower == n_gridpoints - 1 )
			low = &nuclide_grids[nuc*n_gridpoints + lower - 1];
		else
			low = &nuclide_grids[nuc*n_gridpoints + lower];
	}
	
	high = low + 1;
	
	// calculate the re-useable interpolation factor
	f = (high->energy - p_energy) / (high->energy - low->energy);

	// Total XS
	xs_vector[0] = high->total_xs - f * (high->total_xs - low->total_xs);
	
	// Elastic XS
	xs_vector[1] = high->elastic_xs - f * (high->elastic_xs - low->elastic_xs);
	
	// Absorbtion XS
	xs_vector[2] = high->absorbtion_xs - f * (high->absorbtion_xs - low->absorbtion_xs);
	
	// Fission XS
	xs_vector[3] = high->fission_xs - f * (high->fission_xs - low->fission_xs);
	
	// Nu Fission XS
	xs_vector[4] = high->nu_fission_xs - f * (high->nu_fission_xs - low->nu_fission_xs);
	
	//test
	/*
	if( omp_get_thread_num() == 0 )
	{
		printf("Lookup: Energy = %lf, nuc = %d\n", p_energy, nuc);
		printf("e_h = %lf e_l = %lf\n", high->energy , low->energy);
		printf("xs_h = %lf xs_l = %lf\n", high->elastic_xs, low->elastic_xs);
		printf("total_xs = %lf\n\n", xs_vector[1]);
	}
	*/
	
}

// Calculates macroscopic cross section based on a given material & energy 
void calculate_macro_xs( double p_energy, int mat, long n_isotopes,
                         long n_gridpoints, int *  num_nucs,
                         double *  concs,
                         double *  egrid, int *  index_data,
                         NuclideGridPoint *  nuclide_grids,
                         int *  mats,
                         double *  macro_xs_vector, int grid_type, int hash_bins, int max_num_nucs ){
	int p_nuc; // the nuclide we are looking up
	long idx = -1;	
	double conc; // the concentration of the nuclide in the material

	// cleans out macro_xs_vector
	for( int k = 0; k < 5; k++ )
		macro_xs_vector[k] = 0;

	// If we are using the unionized energy grid (UEG), we only
	// need to perform 1 binary search per macroscopic lookup.
	// If we are using the nuclide grid search, it will have to be
	// done inside of the "calculate_micro_xs" function for each different
	// nuclide in the material.
	if( grid_type == UNIONIZED )
		idx = grid_search( n_isotopes * n_gridpoints, p_energy, egrid);	
	else if( grid_type == HASH )
	{
		double du = 1.0 / hash_bins;
		idx = p_energy / du;
	}
	
	// Once we find the pointer array on the UEG, we can pull the data
	// from the respective nuclide grids, as well as the nuclide
	// concentration data for the material
	// Each nuclide from the material needs to have its micro-XS array
	// looked up & interpolatied (via calculate_micro_xs). Then, the
	// micro XS is multiplied by the concentration of that nuclide
	// in the material, and added to the total macro XS array.
	// (Independent -- though if parallelizing, must use atomic operations
	//  or otherwise control access to the xs_vector and macro_xs_vector to
	//  avoid simulataneous writing to the same data structure)
	for( int j = 0; j < num_nucs[mat]; j++ )
	{
		double xs_vector[5];
		p_nuc = mats[mat*max_num_nucs + j];
		conc = concs[mat*max_num_nucs + j];
		calculate_micro_xs( p_energy, p_nuc, n_isotopes,
		                    n_gridpoints, egrid, index_data,
		                    nuclide_grids, idx, xs_vector, grid_type, hash_bins );
		for( int k = 0; k < 5; k++ )
			macro_xs_vector[k] += xs_vector[k] * conc;
	}
	
	//test
	/*
	for( int k = 0; k < 5; k++ )
		printf("Energy: %lf, Material: %d, XSVector[%d]: %lf\n",
		       p_energy, mat, k, macro_xs_vector[k]);
			   */
}


// binary search for energy on unionized energy grid
// returns lower index
long grid_search( long n, double quarry, double *  A)
{
	long lowerLimit = 0;
	long upperLimit = n-1;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );
		
		if( A[examinationPoint] > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;
		
		length = upperLimit - lowerLimit;
	}
	
	return lowerLimit;
}

// binary search for energy on nuclide energy grid
long grid_search_nuclide( long n, double quarry, NuclideGridPoint * A, long low, long high)
{
	long lowerLimit = low;
	long upperLimit = high;
	long examinationPoint;
	long length = upperLimit - lowerLimit;

	while( length > 1 )
	{
		examinationPoint = lowerLimit + ( length / 2 );
		
		if( A[examinationPoint].energy > quarry )
			upperLimit = examinationPoint;
		else
			lowerLimit = examinationPoint;
		
		length = upperLimit - lowerLimit;
	}
	
	return lowerLimit;
}

// picks a material based on a probabilistic distribution
int pick_mat( uint64_t * seed )
{
	// I have a nice spreadsheet supporting these numbers. They are
	// the fractions (by volume) of material in the core. Not a 
	// *perfect* approximation of where XS lookups are going to occur,
	// but this will do a good job of biasing the system nonetheless.

	// Also could be argued that doing fractions by weight would be 
	// a better approximation, but volume does a good enough job for now.

	double dist[12];
	dist[0]  = 0.140;	// fuel
	dist[1]  = 0.052;	// cladding
	dist[2]  = 0.275;	// cold, borated water
	dist[3]  = 0.134;	// hot, borated water
	dist[4]  = 0.154;	// RPV
	dist[5]  = 0.064;	// Lower, radial reflector
	dist[6]  = 0.066;	// Upper reflector / top plate
	dist[7]  = 0.055;	// bottom plate
	dist[8]  = 0.008;	// bottom nozzle
	dist[9]  = 0.015;	// top nozzle
	dist[10] = 0.025;	// top of fuel assemblies
	dist[11] = 0.013;	// bottom of fuel assemblies
	
	double roll = LCG_random_double(seed);

	// makes a pick based on the distro
	for( int i = 0; i < 12; i++ )
	{
		double running = 0;
		for( int j = i; j > 0; j-- )
			running += dist[j];
		if( roll < running )
			return i;
	}

	return 0;
}

double LCG_random_double(uint64_t * seed)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	const uint64_t a = 2806196910506780709ULL;
	const uint64_t c = 1ULL;
	*seed = (a * (*seed) + c) % m;
	return (double) (*seed) / (double) m;
	//return ldexp(*seed, -63);

}	

uint64_t fast_forward_LCG(uint64_t seed, uint64_t n)
{
	// LCG parameters
	const uint64_t m = 9223372036854775808ULL; // 2^63
	uint64_t a = 2806196910506780709ULL;
	uint64_t c = 1ULL;

	n = n % m;

	uint64_t a_new = 1;
	uint64_t c_new = 0;

	while(n > 0) 
	{
		if(n & 1)
		{
			a_new *= a;
			c_new = c_new * a + c;
		}
		c *= (a + 1);
		a *= a;

		n >>= 1;
	}

	return (a_new * seed + c_new) % m;

}
