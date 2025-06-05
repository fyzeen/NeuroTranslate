#!/bin/sh

#   dual_regression - take group-ICA maps (etc) and get subject-specific versions of them (and associated timecourses)
#
#   Stephen Smith, Christian Beckmann, Janine Bijsterbosch, Sam Harrison, FMRIB Image Analysis Group
#
#   Copyright (C) 2011-2019 University of Oxford
#
#   Part of FSL - FMRIB's Software Library
#   http://www.fmrib.ox.ac.uk/fsl
#   fsl@fmrib.ox.ac.uk
#
#   Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
#   Imaging of the Brain), Department of Clinical Neurology, Oxford
#   University, Oxford, UK
#
#
#   LICENCE
#
#   FMRIB Software Library, Release 6.0 (c) 2018, The University of
#   Oxford (the "Software")
#
#   The Software remains the property of the Oxford University Innovation
#   ("the University").
#
#   The Software is distributed "AS IS" under this Licence solely for
#   non-commercial use in the hope that it will be useful, but in order
#   that the University as a charitable foundation protects its assets for
#   the benefit of its educational and research purposes, the University
#   makes clear that no condition is made or to be implied, nor is any
#   warranty given or to be implied, as to the accuracy of the Software,
#   or that it will be suitable for any particular purpose or for use
#   under any specific conditions. Furthermore, the University disclaims
#   all responsibility for the use which is made of the Software. It
#   further disclaims any liability for the outcomes arising from using
#   the Software.
#
#   The Licensee agrees to indemnify the University and hold the
#   University harmless from and against any and all claims, damages and
#   liabilities asserted by third parties (including claims for
#   negligence) which arise directly or indirectly from the use of the
#   Software or the sale of any products based on the Software.
#
#   No part of the Software may be reproduced, modified, transmitted or
#   transferred in any form or by any means, electronic or mechanical,
#   without the express permission of the University. The permission of
#   the University is not required if the said reproduction, modification,
#   transmission or transference is done without financial return, the
#   conditions of this Licence are imposed upon the receiver of the
#   product, and all original and amended source code is included in any
#   transmitted product. You may be held legally responsible for any
#   copyright infringement that is caused or encouraged by your failure to
#   abide by these terms and conditions.
#
#   You are not permitted under this Licence to use this Software
#   commercially. Use for which any financial return is received shall be
#   defined as commercial use, and includes (1) integration of all or part
#   of the source code or the Software into a product for sale or license
#   by or on behalf of Licensee to third parties or (2) use of the
#   Software or any derivative of it for research with the final aim of
#   developing software products for sale or license to a third party or
#   (3) use of the Software or any derivative of it for research with the
#   final aim of developing non-software products for sale or license to a
#   third party, or (4) use of the Software to provide any service to an
#   external organisation for which payment is received. If you are
#   interested in using the Software commercially, please contact Oxford
#   University Innovation ("OUI"), the technology transfer company of the
#   University, to negotiate a licence. Contact details are:
#   fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.
export LC_ALL=C

export LC_ALL=C

Usage() {
    cat <<EOF

dual_regression v0.6

***NOTE*** ORDER OF COMMAND-LINE ARGUMENTS IS DIFFERENT FROM PREVIOUS VERSION

Usage: dual_regression <group_IC_maps> <des_norm> <design.mat> <design.con> <n_perm> [--thr] <output_directory> <input1> <input2> <input3> .........
e.g.   dual_regression groupICA.gica/groupmelodic.ica/melodic_IC 1 design.mat design.con 500 0 grot \`cat groupICA.gica/.filelist\`

<group_IC_maps_4D>            4D image containing spatial IC maps (melodic_IC) from the whole-group ICA analysis
<des_norm>                    0 or 1 (1 is recommended). Whether to variance-normalise the timecourses used as the stage-2 regressors
<design.mat>                  Design matrix for final cross-subject modelling with randomise
<design.con>                  Design contrasts for final cross-subject modelling with randomise
<n_perm>                      Number of permutations for randomise; set to 1 for just raw tstat output, set to 0 to not run randomise at all.
[--thr]                       Perform thresholded dual regression to obtain unbiased timeseries for connectomics analyses (e.g., with FSLnets)
<output_directory>            This directory will be created to hold all output and logfilesg
<input1> <input2> ...         List all subjects' preprocessed, standard-space 4D datasets

<design.mat> <design.con>     can be replaced with just
-1                            for group-mean (one-group t-test) modelling.
If you need to add other randomise options then edit the line after "EDIT HERE" in the dual_regression script

EOF
    exit 1
}

############################################################################

[ "$6" = "" ] && Usage

ORIG_COMMAND=$*

ICA_MAPS=`${FSLDIR}/bin/remove_ext $1` ; shift
echo "ICA maps are in ${ICA_MAPS}"
DES_NORM=--des_norm
if [ $1 = 0 ] ; then
  DES_NORM=""
fi ; shift

if [ $1 = "-1" ] ; then
  DESIGN="-1"
  shift
else
  dm=$1
  dc=$2
  DESIGN="-d $1 -t $2"
  shift 2
fi

NPERM=$1 ; shift

NAF2=0
if [ $1 = "--thr" ] ; then
  NAF2=1
  shift
fi


# NOTES: replaced ID_dr*='<job submit code>' syntax with '<job submit code>' (problematic)

# FSLurmDIR="/scratch/tyoeasley/WAPIAW3/brainrep_data/ICA_data/FSL_slurm"
FSLurmDIR="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/ICA_data/FSL_slurm" 
OUTPUT=`${FSLDIR}/bin/remove_ext $1` ; shift

while [ "_${1}" != _ ] ; do
  INPUTS="$INPUTS `${FSLDIR}/bin/remove_ext $1`"
  shift
done

############################################################################
############################################################################
############################################################################

if ! test -d $OUTPUT
then
	mkdir -p $OUTPUT
	chmod 770 $OUTPUT
fi
LOGDIR="${OUTPUT}/scripts+logs"
if test -d $LOGDIR
then
	rm -rf $LOGDIR
fi
mkdir $LOGDIR
chmod -R 770 $LOGDIR

echo $ORIG_COMMAND > $LOGDIR/command
if [ "$DESIGN" != -1 ] ; then
  /bin/cp $dm $OUTPUT/design.mat
  /bin/cp $dc $OUTPUT/design.con
fi

# Key sizes
N_ICs=`$FSLDIR/bin/fslnvols $ICA_MAPS`

############################################################################
# printf "\ncreating common mask... (part B)\n"
# cat <<EOF > ${LOGDIR}/drB
# \$FSLDIR/bin/fslmerge -t ${OUTPUT}/maskALL \`\$FSLDIR/bin/imglob ${OUTPUT}/mask_*\`
# \$FSLDIR/bin/fslmaths $OUTPUT/maskALL -Tmin $OUTPUT/mask
# EOF
# chmod a+x "${LOGDIR}/drB"
# $FSLurmDIR/fsl_sub_slurm.sh -j true -T 5 -N maskgen2 -l $LOGDIR -f "${LOGDIR}/drB"
# printf "\ncommon mask successfully created!\n\n"

############################################################################
echo "submitting dual regression jobs..."
scratch_root="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch"
echo "inputs ${#INPUTS}"
for i in $INPUTS ; do

  # eid=$( /scratch/tyoeasley/WAPIAW3/utils/fpath_to_eid.sh $i )
  eid=$( ${scratch_root}/NeuroTranslate/generate_ICA/utils/fpath_to_eid.sh $i)  
  echo "${FSLDIR}/bin/fsl_glm -i ${i} -d ${ICA_MAPS} -o ${OUTPUT}/dr_stage1_${eid}.txt --demean -m \"\"" > "${LOGDIR}/drC_${eid}"
  $FSLurmDIR/fsl_sub_slurm.sh -T 20 -N dual_regression -l $LOGDIR -f "${LOGDIR}/drC_${eid}"

  # if ! test -f ${OUTPUT}/dr_stage1_${eid}.txt
  # then
  #   # echo "${FSLDIR}/bin/fsl_glm -i ${i} -d ${ICA_MAPS} -o ${OUTPUT}/dr_stage1_${eid}.txt --demean -m ${OUTPUT}/mask" > "${LOGDIR}/drC_${eid}"
	#   echo "${FSLDIR}/bin/fsl_glm -i ${i} -d ${ICA_MAPS} -o ${OUTPUT}/dr_stage1_${eid}.txt --demean -m \"\"" > "${LOGDIR}/drC_${eid}"
  # 	  $FSLurmDIR/fsl_sub_slurm.sh -T 10 -N dual_regression -l $LOGDIR -f "${LOGDIR}/drC_${eid}"
  # fi
done
