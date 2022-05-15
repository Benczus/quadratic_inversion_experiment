#pwd
if [ ! -d "../models" ] ; then
  mkdir ../models
fi
wget -O ../models/ga_inv_value_2D https://github.com/Benczus/quadratic_inversion_experiment/blob/master/ga_inv_value_2D?raw=true
wget -O ../models/ga_inv_value_3D https://github.com/Benczus/quadratic_inversion_experiment/blob/master/ga_inv_value_3D?raw=true
wget -O ../models/mlpmodel3D https://github.com/Benczus/quadratic_inversion_experiment/blob/master/mlpmodel3D?raw=true

