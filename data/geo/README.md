# Country lookup is not bundled

The web process does not call a location service, and this directory does not contain an IP-to-country table.

A table compiled from the five regional registries' delegated-extended statistics files was the intended source. It is not here because the RIPE NCC copyright statement forbids reproducing RIPE NCC materials and making them available to third parties without written permission. The exception that does allow some use also says the materials must not be modified and must be identified as RIPE NCC documents. A compact lookup table is a modified redistribution, so it is not committed. The verbatim terms are in `licenses/`.

The other four registries' statistics files say they are free to download and use, with a liability disclaimer, and they do not ask for an attribution line. That does not override the RIPE NCC restriction. No partial table is shipped either.

## Accuracy limits, if a permitted table is added later

Delegated statistics record the country where a range was first registered, not where a person is. A multinational company, a VPN, a mobile carrier, and a cloud provider can all be registered in one country and used in another. The files also omit many reassignments. They are not a city or region source, and this app does not derive either.

## Refresh

Do not download the registry files into this repo until RIPE NCC has allowed a modified redistribution in writing. There is no build script in the tree for that reason.
