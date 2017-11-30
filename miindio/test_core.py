import pytest
import os
import json
import copy
import xml.etree.ElementTree as ET

xml_path = os.path.join(os.path.dirname(__file__), 'test.xml')

def test_modelfiles():
    from miindio import MiindIO
    io = MiindIO(xml_path)
    assert io.modelfiles == ["aexp.model", "aexpnoa.model"]
