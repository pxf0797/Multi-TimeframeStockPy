#!usr/bin/env python3
#coding: utf-8
#text.py

import sys
import datetime
import os
import re

class text:
    """ files generate shell
    how to use: python txt.py path filename
    shell only have 3 parameters,
    path, file path can use \/ or \\, other file path should not begain with \/ and(or) \\, otherwise files can not be generated.
    filename, filename can be defined by yourself. """
    __originPath = ''
    __genPath = ''
    __filePath = ''
    __filePathCvt = ''
    __file = ''
    __fileName = ''
    __originSuffix = 'log'
    __convertSuffix = 'txt'
    __encoding = 'gb2312' #'utf-8' 'gb2312' 'gbk'
    __timeFormatFull = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    __timeFormatShort = datetime.datetime.now().strftime('%Y%m%d')#strftime('%y%m%d')
    
    __totatlLines = 0
    __processI = 0
    __startProcess = 0
    
    __combineFlag = False
    __combineStr = ''
    
    def __init__(self,sys_args):
        """ according input to initial

        main function:
        1.generate file name
        2.generate file path """
        if sys_args[1] != '/' and sys_args[1] != '\\':
            self.__genPath = sys_args[1]
        else:
            self.__genPath = ''
        self.__originPath = os.getcwd()
        self.__file = sys_args[2]
        str = self.__file.split('.')
        self.__fileName = str[0]
        self.__originSuffix = str[1]
    
    #======================================================
    # Path
    #------------------------------------------------------
    def generateFilePath(self):
        """ generate file path. """
        if self.__genPath != '':
            if not os.path.exists(self.__genPath):
                os.makedirs(self.__genPath)
            os.chdir(self.__genPath)
    
    def backupOriginPath(self):
        """ back up to original path. """
        if self.__originPath != '':
            os.chdir(self.__originPath)
            
    def getOriginPath(self):
        """ get original path. """
        return self.__originPath
    
    def getGenPath(self):
        """ get generate path. """
        return self.__genPath
    
    def setConvertSuffix(self,suffix):
        """ set convert Suffix. """
        self.__convertSuffix = suffix
    
    def updateFilePath(self):
        """ update file path. """
        if self.__genPath != '':
            self.__filePath = self.__originPath + '/' + self.__genPath + '/' + self.__file
            self.__filePathCvt = self.__originPath + '/' + self.__genPath + '/' + self.__fileName + '.' + self.__convertSuffix
        else:
            self.__filePath = self.__originPath + '/' + self.__file
            self.__filePathCvt = self.__originPath + '/' + self.__fileName + '.' + self.__convertSuffix
    
    def getFilePath(self):
        """ get file path. """
        return self.__filePath
        
    def getFilePathCvt(self):
        """ get convert file path. """
        return self.__filePathCvt    
    
    def setFileName(self,fileName):
        """ set file name. """
        self.__fileName = fileName
    
    def getFileName(self):
        """ get file name. """
        return self.__fileName
    
    #======================================================
    # File format
    #------------------------------------------------------
    def setEncoding(self,encoding):
        """ set encoding. """
        self.__encoding = encoding
    
    def getEncoding(self):
        """ get encode. """
        return self.__encoding
    
    #======================================================
    # Time Format
    #------------------------------------------------------
    def getTimeFormatFull(self):
        """ get time format full. """
        return self.__timeFormatFull
        
    def getTimeFormatShort(self):
        """ get time format short. """
        return self.__timeFormatShort
    
    #======================================================
    # Percent process show
    #------------------------------------------------------
    def setTotalLines(self, totalLines):
        """ set toatl lines and set start process flag, reset process i. """
        self.__totatlLines = totalLines
        self.__startProcess = 1
        self.__processI = 0
    
    def percentShow(self):
        """ show total lines and process percent. """
        # show total lines
        if (self.__startProcess == 1):
            self.__startProcess = 0
            print('total lines: %d' %self.__totatlLines)
        
        # show process percent
        if (self.__processI < self.__totatlLines):
            self.__processI = (self.__processI+1)
            if ((self.__processI&0xF) == 0):
                percent = ((self.__processI*100.0)/self.__totatlLines)
                print('%6.2f%%  %d' %(percent,self.__processI))
    
    #======================================================
    # String operation
    #------------------------------------------------------
    def strEqual(self,inputStr,compareStr):
        ''' inputStr need to strip space on head or end
        compareStr must have not space at head or end.
        '''
        if inputStr.strip() != compareStr:
            return False
        else:
            return True
    
    def str2list(self,inputStr,reExp):
        ''' convert str to list with split of space.
        '''
        str_list = []
        try:
            #print(reExp)
            str_list = re.split(reExp,inputStr)
        except:
            print("str_list format error!!!  -> %s" %inputStr)
        return str_list
    
    def list2str(self,inputList,sperate):
        ''' conbine list to string '''
        str_list = ''
        if (len(inputList)>0):
            for slice_s in inputList:
                if (slice_s != ''):
                    str_list += (slice_s+sperate)
        return str_list
    
    #======================================================
    # Lines operation
    #------------------------------------------------------
    def getCombineFlag(self):
        return self.__combineFlag
    
    def resetCombineFlag(self):
        self.__combineFlag = False
    
    def combineLine(self,line,startStr,endStr):
        ''' combine lines into one with prescribed start and end str
        '''
        try:
            #print(line)
            if (self.__combineFlag):
                # check endStr, end flag
                if self.strEqual(line.strip()[-len(endStr):],endStr):
                    self.__combineFlag = False
                    self.__combineStr += line
                else:
                    # continue to find the end str
                    self.__combineStr += line
            else:
                # if start and end str in line directly return
                if self.strEqual(line.strip()[-len(endStr):],endStr):
                    # find the end str, return line
                    if self.strEqual(line.strip()[0:len(startStr)],startStr):
                        self.__combineStr = line
                    else:
                        # no start str, reset __combineStr
                        self.__combineStr = ''
                else:
                    # find the start str, set flag
                    if self.strEqual(line.strip()[0:len(startStr)],startStr):
                        self.__combineFlag = True
                        self.__combineStr = line
                    else:
                        # no start and end str, reset __combineStr
                        self.__combineStr = ''
        except:
            print("str_list format error!!!  -> %s" %inputStr)
        # if current under combing, return null
        if (self.__combineFlag):
            return ''
        else:
            return self.__combineStr
    
    #======================================================
    # Read And Write
    #------------------------------------------------------
    def openFileReadLines(self,filePath):
        """ read every line in file. """
        try:
            fRead = open(filePath,mode = 'r',encoding=self.__encoding)
            fReadLines = fRead.readlines()
            fRead.close()
        except IOError as e:
            print( "Can't Open File!!!" + filePath, e)
        return fReadLines
    
    def openFileWriteLines(self,filePath,fileData):
        """ if data is not null, then write data in the file. """
        try:
            if (fileData != ''):
                fWrite = open(filePath,mode = 'w',encoding=self.__encoding)
                
                fWrite.write(fileData)
                fWrite.close()
        except IOError as e:
            print( "Can't Open File!!!" + filePath, e)
    
    #======================================================
    # File Operation
    #------------------------------------------------------
    def readLines(self):
        """" read file lines. """
        self.updateFilePath()
        return self.openFileReadLines(self.__filePath)
    
    def writeLines(self,fileData):
        """" wirte file lines. """
        self.updateFilePath()
        self.openFileWriteLines(self.__filePathCvt,fileData)
