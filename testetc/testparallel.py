from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
name=MPI.Get_processor_name()
senddata = (comm.rank+1)*np.arange(comm.size, dtype=int)
recvdata = np.empty(comm.size**2, dtype=int)
comm.Allgather(senddata, recvdata)
print(("name",name,"rank",comm.rank,"own",senddata,"total",recvdata))