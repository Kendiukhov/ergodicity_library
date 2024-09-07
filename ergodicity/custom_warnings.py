"""
Custom Warnings Module Overview

The **`custom_warnings`** module defines a set of custom warning classes that are specifically tailored for the development and use of the library. These warnings are meant to inform users and developers about the status, reliability, and recommended usage of certain features or parameters within the library.

Purpose:

This module is primarily used to alert users when they are interacting with:

- Features that are still under development.

- Functions or parameters that require careful attention.

- Code that has not been thoroughly tested or validated.

By providing specific, tailored warnings, the module enhances the transparency and robustness of the library, ensuring that users are aware of potential issues or limitations when using experimental or edge-case features.

Key Warnings Defined:

1. **InDevelopmentWarning**:

   - **Purpose**: Indicates that a feature is still in development and may not work as intended.

   - **Usage**: Used to alert users when they are interacting with a feature that is under active development and may have limited functionality or unexpected behavior.

   - **Message**: "This feature is still in development and may not function as intended. Please use with caution."

2. **KnowWhatYouDoWarning**:

   - **Purpose**: Warns users that they must understand what they are doing, as the function may not raise an error but could behave unexpectedly with invalid inputs.

   - **Usage**: Useful in cases where the function's behavior depends heavily on correct inputs, and failure to provide valid input could lead to silent failures.

   - **Message**: "Make sure you know what you are doing. This function may not work properly if you do not provide a valid input but will not raise an error."

3. **NotTestedWarning**:

   - **Purpose**: Indicates that a feature has not been thoroughly tested and may have unverified behavior.

   - **Usage**: Warns users that they are using a feature that could behave unpredictably due to a lack of testing or validation.

   - **Message**: "This feature has not been tested and may not function as intended. Please use with caution."

4. **NotRecommendedWarning**:

   - **Purpose**: Alerts users that a specific feature or set of parameters is not recommended for use.

   - **Usage**: Applied when certain configurations or uses of the library are discouraged, even though they might technically work, to prevent misuse or suboptimal performance.

   - **Message**: "This feature or set of parameters is not recommended for use. Please use with caution."

When to Use These Warnings:

These custom warnings should be invoked when:

- Users are interacting with experimental features.

- Developers are working on a function that hasn't yet been validated or widely tested.

- A certain set of inputs or parameters might lead to suboptimal or incorrect behavior without explicitly raising an error.

By using these warnings, both developers and users are encouraged to carefully consider the use of certain features, promoting a more stable and reliable interaction with the library.

"""

import warnings

class InDevelopmentWarning(UserWarning):
    def __init__(self, message):
        # Call the base class constructor
        super().__init__(message)
        # Print the custom message
        print("In Development Warning: This feature is still in development and may not function as intended. Please use with caution.")
        pass

class KnowWhatYouDoWarning(UserWarning):
    def __init__(self, message):
        # Call the base class constructor
        super().__init__(message)
        # Print the custom message
        print("Make Sure You Know What You Are Doing Warning: Make sure you know what you are doing. This function may not work properly if you do not provide a valid input but will not raise an error.")
        pass

class NotTestedWarning(UserWarning):
    def __init__(self, message):
        # Call the base class constructor
        super().__init__(message)
        # Print the custom message
        print("Not Tested Warning: This feature has not been tested and may not function as intended. Please use with caution.")
        pass

class NotRecommendedWarning(UserWarning):
    def __init__(self, message):
        # Call the base class constructor
        super().__init__(message)
        # Print the custom message
        print("Not Recommended Warning: This feature or set of parameters is not recommended for use. Please use with caution.")

