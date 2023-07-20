import tkinter as tk
from tkinter.ttk import Button, OptionMenu, Label
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import matplotlib.dates as mdates
import matplotlib.units as munits
import os
import pytz

from astropy.table import Table
import numpy as np
from scipy.interpolate import interp1d

import heartpy as hp



class EmpaticaData:
    def __init__(self):    
        self.time = None
        self.data = None
        self.sample_freq = None
        self.N = 0
        self.attrib = {}
        
    def add_attrib(self, name, value):
        self.attrib[name] = value
        

class EmpaticaSession:
    def __init__(self, directoryname=None):
        if directoryname is not None:
            self.read_data(directoryname)
            
    def read_data(self, directoryname):
        self.EDA = EmpaticaSession.read_csv_1(f'{directoryname}/EDA.csv')
        self.BVP = EmpaticaSession.read_csv_1(f'{directoryname}/BVP.csv', scale=0.01, optional=True) #?
        self.accel = EmpaticaSession.read_csv_3(f'{directoryname}/ACC.csv', scale=1./64, optional=True)
        self.accel.data -= 1.0
        self.HR = EmpaticaSession.read_csv_1(f'{directoryname}/HR.csv')
        self.temp = EmpaticaSession.read_csv_1(f'{directoryname}/TEMP.csv')
        self.IBI = EmpaticaSession.read_IBI(f'{directoryname}/IBI.csv', optional=True)
#         self.HRV = self.calculate_HRV_from_IBI()
#         self.HRV = self.calculate_HRV_from_BVP()
        self.HRV = self.calculate_HRV_from_IBIBVP()
        self.tags = EmpaticaSession.read_tags(f'{directoryname}/tags.csv')
        self.data_sources = ['EDA','BVP','accel','HR','temp','HRV']
        
    def __getitem__(self, key):
        if key in self.data_sources:
            return getattr(self, key)
        else:
            raise IndexError
        

        
    @staticmethod
    def read_csv_1(filename, scale=1.0, optional=False):
        # Read CSV file with only one column
        EData = EmpaticaData()
        
        try:
            csv = Table.read(filename, format='ascii.csv')
            time0 = float(csv.colnames[0])
            EData.sample_freq = csv[0][0]
            EData.data = csv.columns[0].data[1:] * scale
            EData.N = len(EData.data)
            
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x) for x in time0 + np.arange(EData.N) / EData.sample_freq])
            
        except:
            if optional:
                EData.time = np.array([])
                EData.data = np.array([])
                EData.N = 0
            else:
                raise
        
        return EData
    
    @staticmethod    
    def read_csv_3(filename, scale=1.0, optional=False):
        # Read 3D CSV file and return magnitude
        EData = EmpaticaData()
        
        try:
            csv = Table.read(filename, format='ascii.csv')
            time0 = float(csv.colnames[0])
            EData.sample_freq = csv[0][0]
            
            # We could imagine either taking a specific axis, or looking at the total
            # magnitude. The Empatica Connect interface appears to take the x axis value,
            # but y appears to be the one that shows the most interesting variation.
            # If we wanted the total magnitude, uncomment this instead:
            #EData.data = np.sqrt(csv.columns[0].data[1:]**2 + csv.columns[1].data[1:]**2 +
            #    csv.columns[2].data[1:]**2) * scale
            EData.data = csv.columns[1].data[1:] * scale
            EData.N = len(EData.data)
            
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x) for x in time0 + np.arange(EData.N) / EData.sample_freq])

        except:
            if optional:
                EData.time = np.array([])
                EData.data = np.array([])
                EData.N = 0
            else:
                raise
        
        return EData
    
    @staticmethod    
    def read_tags(filename):
        # Read tag data which is just list of timestamps
        EData = EmpaticaData()
        
        # Tags don't always exist. Check first.
        if os.path.getsize(filename)==0:
            EData.time = []
            EData.N = 0
            EData.data = []
        
        else:
            filedata = Table.read(filename, format='ascii.no_header')
        
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x) for x in filedata.columns[0].data])
            EData.N = len(EData.time)
            EData.data = [1]*EData.N
        
        return EData
        
    @staticmethod
    def read_IBI(filename, optional=True, nan=False):
        EData = EmpaticaData()

        try:
            csv = Table.read(filename, format='ascii.csv')
            if len(csv)==0:
                # equivalent to IOError
                raise IOError
            time0 = float(csv.colnames[0])
            csv.columns[0].name = 't'
        
            # Stick in NaNs for missing beats if requested
            Ngoodbeats = len(csv)
            timelist = [csv['t'][0]]
            beatlist = [csv['IBI'][0]]
            
            for i in range(1,Ngoodbeats):
                if (not nan) or (csv['t'][i] == csv['t'][i-1] + csv['IBI'][i]):
                    # Beat is good -- just include it as is
                    timelist.append(csv['t'][i])
                    beatlist.append(csv['IBI'][i])
                else:
                    # Insert as many NaNs as required first
                    missing_timeperiod = csv['t'][i] - csv['t'][i-1]
                    missing_numberbeats = int(round(missing_timeperiod / csv['IBI'][i]))
                    avg_beatinterval = missing_timeperiod / missing_numberbeats
                    missing_timelist = list(csv['t'][i-1] + np.arange(1,missing_numberbeats+1,1,dtype=int)*avg_beatinterval)
                    missing_beatlist = [np.nan]*missing_numberbeats
                    timelist.extend(missing_timelist)
                    beatlist.extend(missing_beatlist)
                
                    # Then add the final good beat
                    timelist.append(csv['t'][i])
                    beatlist.append(csv['IBI'][i])
                
            # Convert IBI to ms, store
            EData.data = np.array(beatlist)*1e3
            EData.N = len(beatlist)
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x+time0) for x in timelist])
            # Also store other attributes: time in seconds, t0
            EData.add_attrib('t0', time0)
            EData.add_attrib('timelist', np.array(timelist))
        
        except IOError:
            if optional:
                EData.data = np.array([])
                EData.time = np.array([])
                EData.N = 0
                EData.add_attrib('time0', 0.)
                EData.add_attrib('timelist', np.array([]))
            else:
                raise
        
        return EData
        
    def calculate_HRV_from_IBI(self):
        EData = EmpaticaData()

# This code takes the existing IBI intervals from the E4.
        if self.IBI.N > 0:
            segment_nanalyze = 6 # number of segments
            segment_length = 15. # seconds per unit
            nv_thresh = 10  # don't use if there are not at least this many valid intervals
            
            # Find time borders
            tmax = self.IBI.attrib['timelist'][-1] + 1.
            tboundaries = np.arange(0, tmax+segment_nanalyze*segment_length, segment_length)
            IBI_tbounds = np.searchsorted(self.IBI.attrib['timelist'], tboundaries)
            nseg = len(IBI_tbounds)
            
            timelist = []
            hflist = []
            nvalid = []
            for i in range(nseg-segment_nanalyze):
                IBIslice = slice(IBI_tbounds[i], IBI_tbounds[i+segment_nanalyze])
                wd = {'RR_list_cor': self.IBI.data[IBIslice]}
                wd, m = hp.analysis.calc_fd_measures(working_data=wd, method='welch')
                timelist.append(tboundaries[i]+0.5*segment_nanalyze*segment_length)
                nv = len(self.IBI.data[IBIslice])
                nvalid.append(nv)
                if nv >= nv_thresh:
                    hflist.append(m['hf'])
                else:
                    hflist.append(np.nan)
                
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x+self.IBI.attrib['t0']) for x in timelist])
            EData.data = np.array(hflist)
            EData.N = len(timelist)
            # Cap at 1e4
            EData.data[EData.data > 1e4] = np.nan
            
            EData.add_attrib('nvalid', np.array(nvalid))

        else:
            EData.data = np.array([])
            EData.time = np.array([])
            EData.N = 0
            EData.add_attrib('nvalid', np.array([]))
            
        return EData
            
            
            
    def calculate_HRV_from_BVP(self):
        EData = EmpaticaData()
    
# This code uses HeartPy to calculate the RR intervals.        
        if self.BVP.N > 0:
            segment_length = 90. # seconds
            segment_plotevery = 15. # seconds
            overlap = 1. - (segment_plotevery/segment_length)

            # Flip because BVP dips at beats.
            BVPflip = hp.flip_signal(self.BVP.data)
            
            wd, m = hp.process_segmentwise(BVPflip, sample_rate=self.BVP.sample_freq, \
                calc_freq=True, segment_width=segment_length, segment_overlap=overlap, freq_method='welch', \
                replace_outliers=False)
#            pdb.set_trace()
            
            # If m is empty, did not succeed.
            if 'segment_indices' in m:            
                # Times
                EData.time = np.array([self.BVP.time[mx[0]] + datetime.timedelta(seconds=0.5*segment_length) for mx in m['segment_indices']])
            
                # Cap at 1e4
                EData.data = np.array(m['hf'])
                EData.data[EData.data > 1e4] = np.nan
            
                EData.N = len(EData.time)
                
            else:
                EData.data = np.array([])
                EData.time = np.array([])
                EData. N = 0
        
        else:
            EData.data = np.array([])
            EData.time = np.array([])
            EData.N = 0
            
        return EData
    

    def calculate_HRV_from_IBIBVP(self, fdparms={'method':'welch', 'welch_wsize':30}):
        EData = EmpaticaData()

# This code takes the existing IBI intervals from the E4 and combines them with
# those derived by heartpy from BVP.
        if self.IBI.N > 0:
        
            # augment with BVP. Flip because BVP dips at beats.
            BVPflip = hp.flip_signal(self.BVP.data)
            # Often heartpy's process can't deal with the full data set, so break it into
            # four minute segments to find peaks and combine them.
            bvpseglength = 240 # seconds
            bvpsegoverlap = 0.
            wd_bvp, m_bvp = hp.process_segmentwise(BVPflip, sample_rate=self.BVP.sample_freq, \
                segment_width=bvpseglength, segment_overlap=bvpsegoverlap)
            # Combine the peak lists
            all_bvp_peaktimes = []
            # if BVP got none, keyword won't exist
            if 'peaklist' in wd_bvp:
                for i in range(len(wd_bvp['peaklist'])):
                    time0 = self.BVP.time[m_bvp['segment_indices'][i][0]]
                    ntime = m_bvp['segment_indices'][i][1] - m_bvp['segment_indices'][i][0] + 1
                    t = time0 + np.arange(ntime)*datetime.timedelta(seconds=1./self.BVP.sample_freq)
                    these_peaks = np.array(wd_bvp['peaklist'][i])[wd_bvp['binary_peaklist'][i]==1]
                    these_peaktimes = t[these_peaks]
                    all_bvp_peaktimes.extend(these_peaktimes)
            all_bvp_peaktimes = np.array(all_bvp_peaktimes)       
            
            # Take master list of IBI and augment with all BVP that don't have an IBI within 2 samples
            maxmatch = 2./self.BVP.sample_freq
            bvpibi_combi = np.array(sorted(list(self.IBI.time)+list(all_bvp_peaktimes)))
            successive_gaps = np.array([x.total_seconds() for x in bvpibi_combi[1:]-bvpibi_combi[:-1]])
            # good ones have gaps that are more than maxmatch
            goodcombi = (successive_gaps > maxmatch)
            peaklist = bvpibi_combi[:-1][goodcombi]
            
            # Measure intervals between these
            interval = (peaklist[1:] - peaklist[:-1])/datetime.timedelta(seconds=0.001)
            interval_times = peaklist[1:]
            
            # Get rid of all intervals that are outside
            # interval based on the heartrate
            time0 = datetime.datetime.utcfromtimestamp(self.IBI.attrib['t0'])
            interval_times_s = np.array([x.total_seconds() for x in interval_times-time0])
            HRtimes_s = [x.total_seconds() for x in self.HR.time-time0]
            hrinterp = interp1d(HRtimes_s, self.HR.data, bounds_error=False, fill_value=(self.HR.data[0], self.HR.data[-1],))
            interval_estimate_from_hr = 60000./hrinterp(interval_times_s)
            
            # RMSSD is usually <20%, vast majority of outliers are bad beats
            good_intervals = (interval >= 0.8*interval_estimate_from_hr) * \
                (interval <= 1.2*interval_estimate_from_hr)

            interval = interval[good_intervals]
            interval_timelist = interval_times_s[good_intervals]
            
            # Now do frequency analysis as in calculate_HRV_from_IBI
            segment_nanalyze = 6 # number of segments
            segment_length = 15. # seconds per unit
            nv_thresh = 10  # don't use if there are not at least this many valid intervals
            
            # Find time borders
            tmax = interval_timelist[-1] + 1.
            tboundaries = np.arange(0, tmax+segment_nanalyze*segment_length, segment_length)
            interval_tbounds = np.searchsorted(interval_timelist, tboundaries)
            nseg = len(interval_tbounds)
            
            timelist = []
            hflist = []
            nvalid = []
            for i in range(nseg-segment_nanalyze):
                islice = slice(interval_tbounds[i], interval_tbounds[i+segment_nanalyze])
                wd = {'RR_list_cor': interval[islice]}
                wd, m = hp.analysis.calc_fd_measures(working_data=wd, **fdparms)
                timelist.append(tboundaries[i]+0.5*segment_nanalyze*segment_length)
                nv = len(interval[islice])
                nvalid.append(nv)
                if nv >= nv_thresh:
                    hflist.append(m['hf'])
                else:
                    hflist.append(np.nan)
                
            EData.time = np.array([datetime.datetime.utcfromtimestamp(x+self.IBI.attrib['t0']) for x in timelist])
            EData.data = np.array(hflist)
            # Cap at 1e4
            EData.data[EData.data > 1e4] = np.nan
            EData.N = len(timelist)
            
            # Interpolate over stretches of 1 or 2 consecutive nans
            interpable = np.zeros(EData.N, dtype=bool)
            nanbegin = np.zeros(EData.N, dtype=bool)
            nanend = np.zeros(EData.N, dtype=bool)
            nanbegin[1:] = ~np.isnan(EData.data[:-1]) * np.isnan(EData.data[1:])
            nanend[1:-1] = np.isnan(EData.data[1:-1]) * ~np.isnan(EData.data[2:])
            # just 1 nan
            interpable[1:-1] = nanbegin[1:-1] * nanend[1:-1]
            # this plus the next
            interpable[1:-2] += nanbegin[1:-2] * nanend[2:-1]
            # this plus prev
            interpable[2:-1] += nanbegin[1:-2] * nanend[2:-1]
            # Create interpolation function
            notnan = ~np.isnan(EData.data)
            
            # If there are none, return blank
            if np.sum(notnan) < 2:
                EData.data = np.array([])
                EData.time = np.array([])
                EData.N = 0
                EData.add_attrib('nvalid', np.array([]))
            else:
                xval = np.array(range(EData.N))
                ifunc = interp1d(xval[notnan], EData.data[notnan])
                # Replace
                EData.data[interpable] = ifunc(xval[interpable])
            
                        
                EData.add_attrib('nvalid', np.array(nvalid))

        else:
            EData.data = np.array([])
            EData.time = np.array([])
            EData.N = 0
            EData.add_attrib('nvalid', np.array([]))
            
        return EData
            
    
        

class Application(tk.Frame):
    application_title = "Empatica Data Viewer"
    session_name = ""
    session_data = EmpaticaSession()
    axes = {}
    zoom_ranges = []
    base_zoom = []
    button_down_x = None
    button_down_status = False # keep track of double mouseups
    mousemove_cid = None
    xhighlightpoly = {}
    
    def __init__(self, master=None):
        super().__init__(master)
        
        self.createUI()
        self.pack()
        
        self.axes = {'EDA':None, 'BVP':None, 'accel':None, 'HR':None, 'temp':None}
        
    def createUI(self):
        # grid
        self.master.title(self.get_title())
        
        # Open file and unzoom buttons
        self.topFrame = tk.Frame(self.master)
        self.topFrame.pack(fill='x')
        self.open_btn = Button(self.topFrame, text="Open", command=self.open_file)
        self.open_btn.pack(side=tk.LEFT)
        self.unzoom_btn = Button(self.topFrame, text="Unzoom", command=self.unzoom, state=tk.DISABLED)
        self.unzoom_btn.pack(side=tk.LEFT)
        self.print_btn = Button(self.topFrame, text="Print to File", command=self.print_to_file)
        self.print_btn.pack(side=tk.LEFT)
        self.tzval = tk.StringVar(self.topFrame)
        default_tz = "UTC"
        #tzoptions = pytz.common_timezones
        tzoptions = ['US/Alaska', 'US/Arizona', 'US/Central', 'US/Eastern', 'US/Hawaii', 'US/Mountain', 'US/Pacific', 'UTC']
        self.changetz_menu = OptionMenu(self.topFrame, self.tzval, default_tz, *tzoptions, command=self.change_tz)
        self.tzval.set(default_tz)
        self.changetz_menu.pack(side=tk.LEFT)
        
        # plot area
        self.plotFrame = tk.Frame(self.master)
        self.plotFrame.pack(fill='both', expand=True)
        
        self.plot_figure = mpl.figure.Figure(figsize=(8,10))
        self.plot_canvas = FigureCanvasTkAgg(self.plot_figure, self.plotFrame)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=1)

        self.datelocator = mdates.AutoDateLocator()
        self.dateformatter = mdates.ConciseDateFormatter(self.datelocator, tz=pytz.timezone(self.tzval.get()))
        
        # Time of mouse cursor at bottom
        self.bottomFrame = tk.Frame(self.master)
        self.bottomFrame.pack(side=tk.BOTTOM, fill='x')
        
        self.currenttime_str = tk.StringVar(self.bottomFrame)
        self.currenttime_str.set("")
        self.currenttime_label = Label(self.bottomFrame, textvariable=self.currenttime_str)
        self.currenttime_label.pack(side=tk.LEFT)

        # Set up some dummy axes
        self.setup_axes()
        
    def setup_axes(self):
        mosaic = '''
        A
        B
        C
        D
        E
        F
        '''
        axes = self.plot_figure.subplot_mosaic(mosaic, sharex=True)
        self.axes['EDA'] = axes['A']
        self.axes['BVP'] = axes['B']
        self.axes['accel'] = axes['C']
        self.axes['HR'] = axes['D']
        self.axes['HRV'] = axes['E']
        self.axes['temp'] = axes['F']
        
        self.axes['EDA'].set_ylabel('EDA (µS)')
        self.axes['EDA'].set_xticks([])
        self.axes['BVP'].set_ylabel('BVP')
        self.axes['BVP'].set_xticks([])
        self.axes['accel'].set_ylabel('Accel (g)')
        self.axes['accel'].set_xticks([])
        self.axes['HR'].set_ylabel('HR (BPM)')
        self.axes['HR'].set_xticks([])
        self.axes['HRV'].set_ylabel('HF HRV (ms$^2$)')
        self.axes['HRV'].set_xticks([])
        self.axes['temp'].set_ylabel('Temp (°C)')
        self.axes['temp'].xaxis.set_major_locator(self.datelocator)
        self.axes['temp'].xaxis.set_major_formatter(self.dateformatter)
        
        # Create button callbacks
        self.plot_figure.canvas.mpl_connect('button_press_event', lambda e: self.on_mousedown(e))
        self.plot_figure.canvas.mpl_connect('button_release_event', lambda e: self.on_mouseup(e))
        self.plot_figure.canvas.mpl_connect('motion_notify_event', lambda e: self.on_mouse_selecting(e))
        self.plot_figure.canvas.mpl_connect('axes_enter_event', lambda e: self.on_newtime(e))
        self.plot_figure.canvas.mpl_connect('axes_leave_event', lambda e: self.on_outaxes(e))
        
    def on_mousedown(self, event):
        if event.xdata is not None:
            self.button_down_x = event.xdata
            self.button_down_status = True
            
            # Get new bg for blitting
            self.blit_bg = self.plot_figure.canvas.copy_from_bbox(self.plot_figure.bbox)
        
    def on_mouseup(self, event):
        if self.button_down_status:
            if event.xdata is not None:        
                desired_zoom_lim = sorted([self.button_down_x, event.xdata])
                self.zoom_ranges.append(desired_zoom_lim)
                self.set_lim(desired_zoom_lim)
                self.unzoom_btn['state'] = 'normal'
                
        self.button_down_status = False
        # Remove shading
        if self.xhighlightpoly:
            for graph in self.xhighlightpoly:
                self.xhighlightpoly[graph].remove()
            self.xhighlightpoly = {}

        self.plot_figure.canvas.draw()
        # Get bg for blitting
        self.blit_bg = self.plot_figure.canvas.copy_from_bbox(self.plot_figure.bbox)

            
    def set_lim(self, zoomlim):
        # Experimentation reveals that xdata refers to decimal days since epoch.
        # So we can multiply by 3600*24 to get seconds, and then convert to UTC datetime
        datetime_lim = [datetime.datetime.utcfromtimestamp(x*3600*24) for x in zoomlim]
        
        for ax in self.axes:
            mask = (self.session_data[ax].time >= datetime_lim[0]) & \
                (self.session_data[ax].time <= datetime_lim[1])
            self.line[ax].set_data(self.session_data[ax].time[mask], self.session_data[ax].data[mask])
            self.axes[ax].relim()
            self.axes[ax].autoscale_view()
            self.axes[ax].set_xlim(datetime_lim)
        
        self.plot_figure.canvas.draw()
        # Get bg for blitting
        self.blit_bg = self.plot_figure.canvas.copy_from_bbox(self.plot_figure.bbox)
        
            
            
    def unzoom(self):
        if len(self.zoom_ranges) > 0:
            self.zoom_ranges.pop()
            
        if len(self.zoom_ranges) > 0:
            self.set_lim(self.zoom_ranges[-1])
        else:
            self.unzoom_btn['state'] = tk.DISABLED
            self.set_lim(self.base_zoom)


    def on_mouse_selecting(self, event):
        # Update cursor time
        self.on_newtime(event)

        # Do nothing else if button is up
        if not self.button_down_status:
            return
            
        # Restore background
        self.plot_figure.canvas.restore_region(self.blit_bg)
            
        # Does the polygon already exist?
        if self.xhighlightpoly:
            # modify ranges
            for graph in self.axes.keys():
                xy = self.xhighlightpoly[graph].get_xy()
                # format should be [[x0,ymin], [x0,ymax], [x1,ymax], [x1,ymin], [x0.ymin]]
                # and we want to change x1
                xy[2,0] = event.xdata
                xy[3,0] = event.xdata
                self.xhighlightpoly[graph].set_xy(xy)
        else:
            # draw it
            for graph in self.axes.keys():
                self.xhighlightpoly[graph] = self.axes[graph].axvspan(self.button_down_x, event.xdata, \
                    color='black', alpha=0.2, animated=True)
                    
        # Render the artists
        for graph in self.xhighlightpoly:
            self.axes[graph].draw_artist(self.xhighlightpoly[graph])
        # Copy image to GUI state
        self.plot_figure.canvas.blit(self.plot_figure.bbox)
        
        
        
    def on_newtime(self, event):
        if event.inaxes:
            cursordt = mpl.dates.num2date(event.xdata, tz=pytz.timezone(self.tzval.get()))
            self.currenttime_str.set(cursordt.strftime("%H:%M:%S"))
            
    def on_outaxes(self, event):
        # Clear current cursor time
        self.currenttime_str.set("")


    def update_plots(self):
        self.plot_figure.clf()
        self.zoom_ranges = []
        self.line = {}
        
        self.setup_axes()
        
        self.line['EDA'], = self.axes['EDA'].plot(self.session_data.EDA.time, self.session_data.EDA.data, \
            color='tab:blue')
        self.line['BVP'], = self.axes['BVP'].plot(self.session_data.BVP.time, self.session_data.BVP.data, \
            color='tab:brown')
        self.line['accel'], = self.axes['accel'].plot(self.session_data.accel.time, self.session_data.accel.data, \
            color='tab:purple')
        self.line['HR'], = self.axes['HR'].plot(self.session_data.HR.time, self.session_data.HR.data, \
            color='tab:orange')
        self.line['HRV'], = self.axes['HRV'].plot(self.session_data.HRV.time, self.session_data.HRV.data, \
            color='tab:pink')
        self.line['temp'], = self.axes['temp'].plot(self.session_data.temp.time, self.session_data.temp.data, \
            color='tab:green')
        
        self.draw_tags()
                
        self.base_zoom = self.axes['temp'].get_xlim()
                
        self.plot_figure.canvas.draw()
        
        # Get bg for blitting
        self.blit_bg = self.plot_figure.canvas.copy_from_bbox(self.plot_figure.bbox)

        
    def draw_tags(self):
        for t in self.session_data.tags.time:
            for ax in self.axes:
                self.axes[ax].axvline(x=t, color='red', lw=2)
        
    def open_file(self):
        filepath = tk.filedialog.askdirectory(title='Choose Empatica Data Directory')
        if filepath=='': # cancel
            return
            
        try:
            self.session_data = EmpaticaSession(filepath)
            self.session_name = os.path.basename(os.path.normpath(filepath))
            self.master.title(self.get_title())
            self.update_plots()
        except:
            tk.messagebox.showerror(title='File Load Error', \
                message='Directory does not have readable E4 data files.')

    def print_to_file(self):
        filetypes = [('PDF','*.pdf'), ('PNG','*.png'), ('PS','*.ps'), ('EPS','*.eps'), \
            ('SVG','*.svg')]
        filepath = tk.filedialog.asksaveasfilename(title='Print to File', filetypes=filetypes)
        if filepath=='': # cancel
            return
            
        try:
            self.plot_figure.savefig(filepath)
        except:
            tk.messagebox.showerror(title='File Save Error', message='Could not print to file.')
            
            
    def change_tz(self, newtz):
        self.dateformatter = mdates.ConciseDateFormatter(self.datelocator, tz=pytz.timezone(newtz))
        self.axes['temp'].xaxis.set_major_formatter(self.dateformatter)
        self.plot_figure.canvas.draw()

        # Get bg for blitting
        self.blit_bg = self.plot_figure.canvas.copy_from_bbox(self.plot_figure.bbox)
        
        
    def get_title(self):
        if self.session_name == "":
            return self.application_title
        else:
            return f'{self.application_title} - {self.session_name}'
        
        

        
def main():
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
        
if __name__ == '__main__':
    main()
    