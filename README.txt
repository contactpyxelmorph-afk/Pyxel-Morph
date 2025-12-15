PYXEL MORPH
README (Quick Overview)
-----------------------------
OVERVIEW
Pyxel Morph is a desktop application for game-development graphics.
It converts an input image into a green-shaded preview plus palette and tile-mapping data. When recolored according to the generated instructions, the result reproduces the original image exactly or approximately within strict palette and tile limits.

The tool was originally designed for GB Studio but can be used with any engine that relies on 4-color palettes and 8x8 tiles.
-------------
LICENSE TIERS
-------------
Free tier

Maximum palettes: 3

Maximum input colors: 6

Complex image optimization: not available
-------------------------
Pro tier (€10.99 / month)

Maximum palettes: 7

Maximum input colors: 11

Complex image optimization: available
--------------------------------
Diamond tier (€39.99 / month)

Maximum palettes: 999

Maximum input colors: 999

Complex image optimization: available
------------------------------------
Global limits (all tiers)
-----------------------------------
Only 8x8 pixel tiles are supported

Maximum of 2 priority colors

Image dimensions must be divisible by 8
--------------------------------------
BASIC WORKFLOW
--------------------------------------
Select an input image (PNG recommended, JPG supported)

Select an output folder

Choose processing mode and number of palettes

Generate the green preview

If the image exceeds the limits of the active license tier, processing will require an upgrade.
---------------------------------------
OUTPUT FILES
--------------------------------------
The selected output folder may contain:

--> Green preview image (PNG)

--> Palettes file (JSON, hex color values)

--> Tile-to-palette mapping file (CSV)

--> Optional recolored preview image

These files are used to manually recreate the scene in the target game-development platform.
-------------------------------------------
ACCOUNTS AND SUBSCRIPTIONS
----------------------------------
User registration and login are required

License upgrades may take approximately 50 seconds to apply after startup

Subscriptions are billed monthly

Canceling a subscription stops renewal and terminates the license 30 days after the subscription start date
---------------------------------------
SUPPORT AND LEGAL
-----------------------------------------
Support email: contact.pyxelmorph@gmail.com

Use of Pyxel Morph is subject to the End User License Agreement and Privacy Policy included with this distribution and available in the project repository.