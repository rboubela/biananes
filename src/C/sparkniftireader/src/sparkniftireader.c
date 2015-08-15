
#include <stdio.h>
#include <nifti1_io.h>

typedef int bool;
enum { false, true };


long get_nvoxel(char * fin) {

  nifti_image * nim = NULL;
  long nvoxel = 0;

  nim = nifti_image_read(fin, 0);

  if( !nim ) {
    fprintf(stderr,"** failed to read NIfTI image from '%s'\n", fin);
    return 2;
  }

  nvoxel = nim->nvox;

  nifti_image_free(nim);

  return nvoxel;
}

long get_nvolumes(char * fin) {
  nifti_image * nim = NULL;
  long nvolumes = 0;

  nim = nifti_image_read(fin, 0);

  if( !nim ) {
    fprintf(stderr,"** failed to read NIfTI image from '%s'\n", fin);
    return 2;
  }

  nvolumes = nim->dim[4];
  nifti_image_free(nim);

  return nvolumes;
}

long get_nvoxel_in_mask(char * mask_file) {
  nifti_image * nim_mask = NULL;
  long i = 0, nvoxel = 0;

  nim_mask = nifti_image_read(mask_file, 1);

  if( !nim_mask ) {
    fprintf(stderr,"** failed to read NIfTI image from '%s'\n", mask_file);
    return 2;
  }

  switch(nim_mask->datatype) {
    case NIFTI_TYPE_UINT8:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        if( ((unsigned char*)nim_mask->data)[i] > 0 )
          nvoxel++;
      }
      break;
    case NIFTI_TYPE_INT16:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        if( ((signed short*)nim_mask->data)[i] > 0 )
          nvoxel++;
      }
      break;
    case NIFTI_TYPE_INT32:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        if( ((int*)nim_mask->data)[i] > 0 )
          nvoxel++;
      }
      break;
    case NIFTI_TYPE_FLOAT32:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        if( ((float*)nim_mask->data)[i] > 0 )
          nvoxel++;
      }
      break;
    case NIFTI_TYPE_FLOAT64:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        if( ((double*)nim_mask->data)[i] > 0 )
          nvoxel++;
      }
      break;
    default:
      fprintf(stderr, "** unsupported datatype in NifTI image from '%s'\n", mask_file);
      nifti_image_free(nim_mask);
      return 2;
  }

  nifti_image_free(nim_mask);

  return nvoxel;
}

int larray_nifti_read_masked_brick(double * pLArray, char * fin, char * mask_file, int nbricks, const int * blist) {

  nifti_image * nim = NULL;
  nifti_image * nim_mask = NULL;
  nifti_brick_list NBL;
  bool * mask = NULL;
  long i = 0, j = 0, n_voxel_per_volume_mask, n_voxel_per_volume_input;

  nim = nifti_image_read_bricks(fin, nbricks, blist, &NBL);

  if( !nim ) {
    printf("** failed to read NIfTI image from '%s'\n", fin);
    return 2;
  }

  nim_mask = nifti_image_read(mask_file, 1);

  if( !nim_mask ) {
    printf("** failed to read NIfTI image from '%s'\n", mask_file);
    return 2;
  }

  n_voxel_per_volume_input = nim->dim[1] * nim->dim[2] * nim->dim[3];
  n_voxel_per_volume_mask = nim_mask->dim[1] * nim_mask->dim[2] * nim_mask->dim[3];

  if (n_voxel_per_volume_input != n_voxel_per_volume_mask) {
    printf("** dimensions of mask in input do not match!");
    return 2;
  }

  mask = (bool*)calloc(n_voxel_per_volume_mask, sizeof(bool));

  switch(nim_mask->datatype) {
    case NIFTI_TYPE_UINT8:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        mask[i] = ((unsigned char*)nim_mask->data)[i] > 0 ? true : false;
      }
      break;
    case NIFTI_TYPE_INT16:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        mask[i] = ((signed short*)nim_mask->data)[i] > 0 ? true : false;
      }
      break;
    case NIFTI_TYPE_INT32:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        mask[i] = ((int*)nim_mask->data)[i] > 0 ? true : false;
      }
      break;
    case NIFTI_TYPE_FLOAT32:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        mask[i] = ((float*)nim_mask->data)[i] > 0 ? true : false;
      }
      break;
    case NIFTI_TYPE_FLOAT64:
      for( i = 0; i < nim_mask->nvox; i++ ) {
        mask[i] = ((double*)nim_mask->data)[i] > 0 ? true : false;
      }
      break;
    default:
      fprintf(stderr, "** unsupported datatype in NifTI image from '%s'\n", mask_file);
  }

  j = 0;

  switch(nim->datatype) {
    case NIFTI_TYPE_UINT8:
      for( i = 0; i < nim->nvox; i++ ) {
        if( mask[i % n_voxel_per_volume_mask] ) {
          pLArray[j] = (double)(((unsigned char**)NBL.bricks)[0])[i];
          j++;
        }
      }
      break;
    case NIFTI_TYPE_INT16:
      for( i = 0; i < nim->nvox; i++ ) {
        if( mask[i % n_voxel_per_volume_mask] ) {
          pLArray[j] = (double)(((signed short**)NBL.bricks)[0])[i];
          j++;
        }
      }
      break;
    case NIFTI_TYPE_INT32:
      for( i = 0; i < nim->nvox; i++ ) {
        if( mask[i % n_voxel_per_volume_mask] ) {
          pLArray[j] = (double)(((int**)NBL.bricks)[0])[i];
          j++;
        }
      }
      break;
    case NIFTI_TYPE_FLOAT32:
      for( i = 0; i < nim->nvox; i++ ) {
        if( mask[i % n_voxel_per_volume_mask] ) {
          pLArray[j] = (double)(((float**)NBL.bricks)[0])[i];
          j++;
        }
      }
      break;
    case NIFTI_TYPE_FLOAT64:
      for( i = 0; i < nim->nvox; i++ ) {
        if( mask[i % n_voxel_per_volume_mask] ) {
          pLArray[j] = (double)(((double**)NBL.bricks)[0])[i];
          j++;
        }
      }
      break;
    default:
      fprintf(stderr, "** unsupported datatype in NifTI image from '%s'\n", mask_file);
  }

  nifti_image_free(nim);
  nifti_image_free(nim_mask);
  nifti_free_NBL(&NBL);

  free(mask);

  return 0;
}

