# mpi.sh
# !/bin/sh
# PBS −N mpi
# PBS −l nodes=8
pssh -h $PBS_NODEFILE mkdir -p /home/ss2011748/MPI 1>&2 
scp master:/home/ss2011748/MPI/Gauss /home/ss2011748/MPI
pscp -h $PBS_NODEFILE /home/ss2011748/MPI/Gauss /home/ss2011748/MPI 1>&2
mpiexec -np 8 -machinefile $PBS_NODEFILE /home/ss2011748/MPI/Gauss