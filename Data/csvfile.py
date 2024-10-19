#!usr/bin/env python3
#coding: utf-8
#csvfile.py

import sys
import datetime
import os
import pandas as pd
import text
import csv

class csvfile:
    """ files generate shell
    how to use: python txt.py path filename
    shell only have 3 parameters,
    path, file path can use \/ or \\, other file path should not begain with \/ and(or) \\, otherwise files can not be generated.
    filename, filename can be defined by yourself. """
    __text = ''
    __convertSuffix = 'csv'
    __encoding = 'utf-8' #'utf-8' 'gb2312' ''
    
    __fRead = ''
    __fReadLines = ''
    __fHeader = []
    __fList = []
    __sortData = []
    
    def __init__(self,sys_args):
        """ according input to initial

        main function:
        1.generate file name
        2.generate file path """
        self.__text = text.text(sys_args)
        self.__text.setConvertSuffix(self.__convertSuffix)
        self.__text.setEncoding(self.__encoding)
    
    #======================================================
    # Params
    #------------------------------------------------------
    def setEncoding(self,encoding):
        """ set encoding. """
        self.__encoding = encoding
        
    def getEncoding(self):
        """ get encoding. """
        return self.__encoding
        
    def setFileName(self,fileName):
        """ set file name. """
        self.__text.setFileName(fileName)
    
    def getFileName(self):
        """ get file name. """
        return self.__text.getFileName()
    
    #======================================================
    # Read And Write
    #------------------------------------------------------
    def openFileRead(self,filePath):
        """ read every line in file. """
        #fRead = ''
        #f = ''
        try:
            #fRead = pd.read_csv(filePath,sep=',',header=0)
            #with open(filePath, 'r', encoding=self._encoding) as f:
                #fReadLines = csv.reader(f)
                #print(type(fReadLines))
                #for row in fReadLines:
                #    print(row)
            
            #fReadLines = csv.reader(open(filePath, 'r', encoding=self._encoding))
            self.__fRead = open(filePath, 'r', encoding=self.__encoding)
            self.__fReadLines = csv.reader(self.__fRead)
        except IOError as e:
            print( "Can't Open File!!!" + filePath, e)
        return self.__fReadLines
    
    def closeFileRead(self):
        """ close open file. """
        self.__fRead.close()
    
    def openFileWrite(self,filePath,fileData):
        """ if data is not null, then write data in the file. """
        try:
            #if (fileData != ''):
                #dataframe = pd.DataFrame(fileData)
                #dataframe.to_csv(filePath,index=False,sep=',')
                #dataframe.to_csv(filePath)
            with open(filePath, 'w', newline='', encoding = self.__encoding) as f:
                writer = csv.writer(f)
                writer.writerows(fileData)
        except IOError as e:
            print( "Can't Open File!!!" + filePath, e)
    
    #======================================================
    # File Operation
    #------------------------------------------------------
    def readLines(self):
        """" read file lines. """
        self.__text.updateFilePath()
        return self.openFileRead(self.__text.getFilePath())
    
    def writeLines(self,fileData):
        """" wirte file lines. fileData must be string."""
        self.__text.updateFilePath()
        self.__text.writeLines(fileData)
        
    def writeLists(self,fileData):
        """" wirte file lines. """
        self.__text.updateFilePath()
        self.openFileWrite(self.__text.getFilePathCvt(),fileData)
        
    #======================================================
    # File Datas Operation
    #------------------------------------------------------
    def change2list(self):
        ''' change file to list.
        split header and data list.
        and delete empty line.'''
        self.__fHeader = next(self.__fReadLines)
        self.__fList = list(self.__fReadLines)
        for i in range(len(self.__fList)):
            for data in self.__fList[i]:
                if (data != ''):
                    self.__sortData.append(self.__fList[i])
                    break
    
    def getHeader(self):
        ''' get header '''
        return self.__fHeader
    
    def getDatas(self):
        ''' get list datas '''
        return self.__sortData
        