"""
ERSIN SQLAlchemy helper functions for type safety
ERSIN These functions help avoid type: ignore comments when using SQLAlchemy methods
"""
from sqlalchemy .sql import ColumnElement 
from typing import cast ,Any 
from sqlalchemy import or_ 


def column_in (column :Any ,values :list [Any ])->Any :
    """
    ERSIN SQLAlchemy column .in_() method wrapper for type safety
    ERSIN Returns a filter expression that can be used in SQLAlchemy queries
    """
    if not values :
    # ERSIN Empty list should return False condition
        return False 
    col =cast (ColumnElement [Any ],column )
    in_method =getattr (col ,'in_',None )
    if in_method is not None and callable (in_method ):
        return in_method (values )
        # ERSIN Fallback: if in_() is not available, create OR conditions
    return or_ (*[col ==val for val in values ])
