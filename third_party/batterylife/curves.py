"""Per-cycle charge/discharge curve resampling, copied from the official BatteryLife repository.

Source: https://github.com/Ruifeng-Tan/BatteryLife, commit 3b17bb53202e10711deae0448c91f84144bb3dd9,
file data_provider/data_loader.py, class Dataset_original, methods read_cell_df, get_charge_discharge_curves and
resample_charge_discharge_curves (MIT License, Copyright (c) 2025 Ruifeng Tan; full text in LICENSE next to this file).

The three methods are copied verbatim except for the lines marked "# [vendor]":
* read_cell_df takes (data, eol, file_name) instead of reading the pkl itself, and drops the data-augmentation call
  (self.aug_helper.batch_aug), whose output is not part of the model input.
* __init__ is reduced to the attributes these methods use.
No processing step was changed.
"""
import numpy as np
import pandas as pd


class BatteryLifeCurves:
    def __init__(self, charge_discharge_length=300, early_cycle_threshold=100):  # [vendor]
        self.ZN_coin_charge_first_file_names = ['ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_430-3_20231212185323_03_2.pkl']
        self.charge_discharge_len = charge_discharge_length  # The resampled length for charge and discharge curves
        self.early_cycle_threshold = early_cycle_threshold
        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']

    def read_cell_df(self, data, eol, file_name):  # [vendor] data and eol are passed in
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, basic_prompt, eol
        '''
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]

        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']

        cycle_data = data['cycle_data'] # list of cycle data dict
        valid_cycle_number = len(cycle_data)

        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah']<0] = np.nan # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah']<0] = np.nan
            cycle_df.bfill(inplace=True) # deal with NaN
            total_cycle_dfs.append(cycle_df)

            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break

        df = pd.concat(total_cycle_dfs)

        # obtain the charge and discahrge curves
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold, nominal_capacity)
        # [vendor] augmentation call removed

        return df, charge_discharge_curves, eol, nominal_capacity, valid_cycle_number  # [vendor] no augmented curves

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []
        unique_cycles = df['cycle_number'].unique()
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle in range(1, early_cycle_threshold+1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]

                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records/nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values
                time_in_s_records = cycle_df['time_in_s'].values

                cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C<=-0.01)
                discharge_end_index = cutoff_voltage_indices[0][-1]

                # tmp_discharge_capacity_records = max(charge_capacity_records) - discharge_capacity_records
                if prefix in ['RWTH', 'CALB_0', 'CALB_25', 'CALB_45'] or (file_name not in self.ZN_coin_charge_first_file_names and prefix=='ZN-coin'):
                    # Every cycle first discharge and then charge
                    #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]

                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity

                    charge_voltages = charge_voltages[np.abs(charge_current_in_C)>0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C)>0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C)>0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C)>0.01]
                else:
                    # Every cycle first charge and then discharge
                    #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity

                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C)>0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C)>0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C)>0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C)>0.01]

                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(charge_voltages, charge_currents, charge_capacities)

                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)

                voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(voltage_records) # normalize using the cutoff voltage
                current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the capacity

                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))

        curves = np.concatenate(curves, axis=0) # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or dicharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interploted records
        '''
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages)+1)
        interp_bases = np.linspace(1, len(voltages)+1, num=charge_discharge_len,
                                        endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery
