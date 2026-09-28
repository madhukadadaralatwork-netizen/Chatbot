"""
app_config.py

Central place to register which documents belong to which application.
When you onboard app #3 or #4, add one line here -- nothing else in the
pipeline needs to change.
"""

APP_DOCUMENTS = {
    "450T": [
        "Technical_Document_450T_AI.docx",
        "450T_User_Guide.xlsx",
    ],
    "MIT": [
        "MIT_Help_Document.docx"
    ]
    # "ProductMatching": [
    #     "ProductMatching_Spec.docx",
    #     "ProductMatching_ColumnRules.xlsx",
    # ],
    # "AppThree": [...],
    # "AppFour": [...],
}