from PyLTSpice import RawRead, SimRunner, SpiceEditor
import numpy as np
import time
import os
import re

#def run_simulation(ton_start=480e-9, ton_end=580e-9, ton_step=10e-9):
def run_simulation(ton_start=480e-9, ton_end=580e-9, ton_step=30e-9, tperiod=1e-6, trise=5e-9, tfall=5e-9):

    #Select LTspice model and create netlist
    LTC = SimRunner(output_folder='./temp')
    LTC.create_netlist('BoostConverter.asc')
    netlist = SpiceEditor('BoostConverter.net')
    #LTC.add_command_line_switch('-r')  # Erzwingt die Erstellung der .raw-Datei
    #LTC.add_command_line_switch('-b')  # Führe die Simulation im Batch-Modus aus (keine GUI)
    LTC.add_command_line_switch('-alt')
    netlist.add_instruction('.options method=Gear')
    #netlist.add_instructions('.step param ton %s %s %s' % (ton_start, ton_end, ton_step))

    # Füge die Parameter über `set_parameter` hinzu
    netlist.set_parameter('ton_start', ton_start)
    netlist.set_parameter('ton_end', ton_end)
    netlist.set_parameter('ton_step', ton_step)
    netlist.set_parameter('tperiod', tperiod)
    netlist.set_parameter('trise', trise)
    netlist.set_parameter('tfall', tfall)

    #LTC.run(netlist)
    LTC.run(netlist, timeout=3600)  # Timeout auf 1200 Sekunden (20 Minuten) setzen
    time.sleep(1)

    #Run a simulation and read data
    AllFiles = os.listdir("./temp")
    count = [None] * len(AllFiles)
    for ii in range(len(AllFiles)):
        FilenamePart = re.split("[_.]", AllFiles[ii])
        count[ii] = FilenamePart[1]
    ActFile = max(count)
    PathFail = 'temp/BoostConverter_%s.fail' % str(ActFile)
    PathRaw = 'temp/BoostConverter_%s.raw' % str(ActFile)
    PathNet = 'temp/BoostConverter_%s.net' % str(ActFile)

    #Check if simulation fails
    if os.path.isfile(PathFail) == False:
        while LTC.wait_completion() == False:
            time.sleep(0.1)
        return PathRaw
    else:
        LTC.updated_stats()
        LTC.kill_all_ltspice()
        os.remove(PathFail)
        os.remove(PathNet)
        os.remove(PathRaw)
        print('Simulation failed')
        return None

def read_simulation_data (path_raw):
    LTR = RawRead(path_raw)
    uds_LT = LTR.get_trace('V(uds)').get_wave()
    t_LT = LTR.get_trace('time').get_wave()
    #IV1_LT = LTR.get_trace('I(V1)').get_wave()
    return t_LT, uds_LT
