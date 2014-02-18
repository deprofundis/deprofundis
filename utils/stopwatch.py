########################################################
# from http://effbot.org/librarybook/timing.htm
# File: timing-example-2.py

"""
This is my wrapper for the time module. There's probably an
easier way to time the duration of things, but when I looked
into timing stuff, this was the best I could come up with...

To use:

    t = Stopwatch()    
    # do something
    elapsed = t.finish() # in milliseconds
"""

import time


class Stopwatch:
    """
    Creates stopwatch timer objects.
    """
    # stores the time the stopwatch was started
    t0 = None
    # stores the time the stopwatch was last looked at
    t1 = None
    
    def __init__(self):
        self.t0 = 0
        self.t1 = 0
        self.start()

    def start(self):
        """
        Stores the current time in t0.
        """
        self.t0 = time.time()

    def finish(self, milli=True):
        """
        Returns the elapsed duration in milliseconds. This
        stores the current time in t1, and calculates the
        difference between t0 (the stored start time) and
        t1, so if you call this multiple times, you'll get a
        larger answer each time.

        You have to call this in order to update t1.
        """        
        self.t1 = time.time()
        if milli:
            return self.milli()
        else:
            return self.seconds()

    def seconds(self):
        """
        Returns t1 - t0 in seconds. Does not update t1.
        """
        return self.t1 - self.t0
        
    def milli(self):
        return int((self.t1 - self.t0) * 1000)
        
