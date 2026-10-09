# autofly_ue5_proto 1.0.0

AutoFly-format UAV navigation episodes (UE 5.7 + Project AirSim), merged from 36 raw stores. Created 2026-10-09 03:55:47; export `data/autofly_ue5_proto/1.0.0`.

| split | episodes | records |
|---|---|---|
| train | 1710 | 322125 |
| val | 90 | 16882 |
| all | 1800 | 339007 |

## Counts

### Scene

| scene | train | val | episodes | records |
|---|---|---|---|---|
| s01 | 191 | 9 | 200 | 38868 |
| s01d | 188 | 12 | 200 | 39572 |
| s02 | 187 | 13 | 200 | 36612 |
| s03 | 192 | 8 | 200 | 39137 |
| s04 | 190 | 10 | 200 | 37813 |
| s05 | 192 | 8 | 200 | 38136 |
| s06 | 191 | 9 | 200 | 37066 |
| s07 | 191 | 9 | 200 | 35249 |
| s09 | 188 | 12 | 200 | 36554 |

### Target

| target | train | val | episodes | records |
|---|---|---|---|---|
| yellow wet floor sign | 49 | 4 | 53 | 9686 |
| man in black suit | 49 | 2 | 51 | 9446 |
| bald man in green shirt | 43 | 4 | 47 | 8720 |
| orange cone | 43 | 4 | 47 | 8736 |
| doctor in white coat | 42 | 3 | 45 | 8467 |
| old television | 42 | 3 | 45 | 8454 |
| girl in yellow shirt | 42 | 2 | 44 | 8427 |
| green armchair | 42 | 2 | 44 | 8311 |
| person in purple shirt | 42 | 2 | 44 | 8000 |
| white cabinet | 40 | 4 | 44 | 8517 |
| yellow cube | 41 | 3 | 44 | 8398 |
| green sphere | 39 | 4 | 43 | 7957 |
| round wooden table | 42 | 1 | 43 | 8118 |
| red sedan | 41 | 1 | 42 | 8172 |
| red cube | 39 | 2 | 41 | 7826 |
| red fire extinguisher | 40 | 1 | 41 | 7861 |
| blue school chair | 38 | 2 | 40 | 7718 |
| bronze whale statue | 38 | 2 | 40 | 7529 |
| man in red shirt | 39 | 1 | 40 | 7353 |
| man in blue jacket | 36 | 3 | 39 | 7173 |
| man in brown jacket | 36 | 3 | 39 | 7196 |
| white plastic chair | 38 | 1 | 39 | 7521 |
| beige sofa | 36 | 2 | 38 | 7211 |
| orange cylinder | 33 | 4 | 37 | 6915 |
| police officer | 37 | 0 | 37 | 7140 |
| red wooden bench | 36 | 1 | 37 | 6893 |
| grey armchair | 34 | 2 | 36 | 6741 |
| black leather sofa | 33 | 1 | 34 | 6442 |
| blue cone | 34 | 0 | 34 | 6384 |
| green suitcase | 34 | 0 | 34 | 6583 |
| red tool chest | 31 | 2 | 33 | 6128 |
| yellow generator | 32 | 1 | 33 | 6180 |
| blue van | 30 | 2 | 32 | 6110 |
| green hatchback | 32 | 0 | 32 | 5791 |
| police car | 30 | 2 | 32 | 6030 |
| blue cylinder | 29 | 2 | 31 | 5956 |
| green garbage truck | 29 | 2 | 31 | 6058 |
| blue barrel | 27 | 2 | 29 | 5511 |
| covered car | 27 | 1 | 28 | 4955 |
| white pickup truck | 27 | 1 | 28 | 5395 |
| wooden barrel | 26 | 2 | 28 | 5171 |
| yellow suv | 27 | 1 | 28 | 5375 |
| red cylinder | 26 | 1 | 27 | 4973 |
| red barrel | 24 | 1 | 25 | 4634 |
| ambulance | 23 | 1 | 24 | 4583 |
| red fire truck | 23 | 1 | 24 | 4691 |
| yellow cylinder | 23 | 1 | 24 | 4333 |
| green cube | 23 | 0 | 23 | 4288 |
| yellow sphere | 21 | 2 | 23 | 4353 |
| yellow taxi | 22 | 1 | 23 | 4597 |

### Category

| category | train | val | episodes | records |
|---|---|---|---|---|
| human | 366 | 20 | 386 | 71922 |
| geometry | 351 | 23 | 374 | 70119 |
| furniture | 339 | 16 | 355 | 67472 |
| vehicle | 311 | 13 | 324 | 61757 |
| equipment | 305 | 16 | 321 | 60208 |
| other | 38 | 2 | 40 | 7529 |

### Seen / unseen

| seen | train | val | episodes | records |
|---|---|---|---|---|
| seen | 1710 | 90 | 1800 | 339007 |

### Stand-in status

| stand_in | train | val | episodes | records |
|---|---|---|---|---|
| not a stand-in | 1060 | 58 | 1118 | 210283 |
| stand-in | 650 | 32 | 682 | 128724 |

### Pool status

| target_status | train | val | episodes | records |
|---|---|---|---|---|
| imported | 1359 | 67 | 1426 | 268888 |
| primitive | 351 | 23 | 374 | 70119 |

### Scene split

| split_of_scene | train | val | episodes | records |
|---|---|---|---|---|
| train | 1710 | 90 | 1800 | 339007 |

### Store

| store | train | val | episodes | records |
|---|---|---|---|---|
| s01_c1 | 49 | 1 | 50 | 9747 |
| s01_c2 | 47 | 3 | 50 | 9699 |
| s01_c3 | 47 | 3 | 50 | 9715 |
| s01_c4 | 48 | 2 | 50 | 9707 |
| s01d_c1 | 46 | 4 | 50 | 9845 |
| s01d_c2 | 48 | 2 | 50 | 9680 |
| s01d_c3 | 45 | 5 | 50 | 10171 |
| s01d_c4 | 49 | 1 | 50 | 9876 |
| s02_c1 | 47 | 3 | 50 | 9258 |
| s02_c2 | 49 | 1 | 50 | 9123 |
| s02_c3 | 45 | 5 | 50 | 8890 |
| s02_c4 | 46 | 4 | 50 | 9341 |
| s03_c1 | 49 | 1 | 50 | 10278 |
| s03_c2 | 47 | 3 | 50 | 9480 |
| s03_c3 | 48 | 2 | 50 | 9713 |
| s03_c4 | 48 | 2 | 50 | 9666 |
| s04_c1 | 47 | 3 | 50 | 9737 |
| s04_c2 | 46 | 4 | 50 | 9494 |
| s04_c3 | 49 | 1 | 50 | 9455 |
| s04_c4 | 48 | 2 | 50 | 9127 |
| s05_c1 | 50 | 0 | 50 | 9681 |
| s05_c2 | 47 | 3 | 50 | 9667 |
| s05_c3 | 46 | 4 | 50 | 9489 |
| s05_c4 | 49 | 1 | 50 | 9299 |
| s06_c1 | 48 | 2 | 50 | 9306 |
| s06_c2 | 50 | 0 | 50 | 9489 |
| s06_c3 | 45 | 5 | 50 | 9192 |
| s06_c4 | 48 | 2 | 50 | 9079 |
| s07_c1 | 48 | 2 | 50 | 8751 |
| s07_c2 | 47 | 3 | 50 | 8836 |
| s07_c3 | 48 | 2 | 50 | 8928 |
| s07_c4 | 48 | 2 | 50 | 8734 |
| s09_c1 | 47 | 3 | 50 | 9031 |
| s09_c2 | 45 | 5 | 50 | 9163 |
| s09_c3 | 49 | 1 | 50 | 9218 |
| s09_c4 | 47 | 3 | 50 | 9142 |

## Instructions

Templates (AutoFly's, verbatim; `{obstacle}` is the scene's phrase):

- `go through and avoid the {obstacle} or other obstacles to reach the {target}`
- `advance to {target} while avioding {obstacle} and other obstacles`

| scene | obstacle phrase |
|---|---|
| s01 | white pillars |
| s01d | white pillars |
| s02 | trees |
| s03 | rocks |
| s04 | stone field |
| s05 | standing stones |
| s06 | tree clusters |
| s07 | stacked boxes |
| s09 | coloured poles |

660 distinct instructions:

| instruction | train | val | episodes | records |
|---|---|---|---|---|
| advance to green sphere while avioding white pillars and other obstacles | 9 | 1 | 10 | 1904 |
| go through and avoid the white pillars or other obstacles to reach the police car | 9 | 1 | 10 | 1923 |
| advance to white plastic chair while avioding white pillars and other obstacles | 9 | 0 | 9 | 1860 |
| go through and avoid the white pillars or other obstacles to reach the old television | 9 | 0 | 9 | 2005 |
| go through and avoid the white pillars or other obstacles to reach the red sedan | 9 | 0 | 9 | 1676 |
| advance to man in blue jacket while avioding white pillars and other obstacles | 8 | 0 | 8 | 1548 |
| advance to person in purple shirt while avioding white pillars and other obstacles | 8 | 0 | 8 | 1440 |
| advance to yellow cube while avioding white pillars and other obstacles | 6 | 2 | 8 | 1453 |
| advance to yellow generator while avioding white pillars and other obstacles | 7 | 1 | 8 | 1542 |
| go through and avoid the white pillars or other obstacles to reach the green garbage truck | 7 | 1 | 8 | 1638 |
| advance to man in black suit while avioding white pillars and other obstacles | 7 | 0 | 7 | 1291 |
| advance to man in red shirt while avioding stone field and other obstacles | 6 | 1 | 7 | 1282 |
| advance to orange cone while avioding white pillars and other obstacles | 7 | 0 | 7 | 1312 |
| advance to yellow taxi while avioding white pillars and other obstacles | 7 | 0 | 7 | 1354 |
| go through and avoid the standing stones or other obstacles to reach the bald man in green shirt | 6 | 1 | 7 | 1305 |
| go through and avoid the white pillars or other obstacles to reach the round wooden table | 7 | 0 | 7 | 1534 |
| go through and avoid the white pillars or other obstacles to reach the white cabinet | 6 | 1 | 7 | 1459 |
| advance to beige sofa while avioding white pillars and other obstacles | 6 | 0 | 6 | 1304 |
| advance to blue school chair while avioding white pillars and other obstacles | 6 | 0 | 6 | 1427 |
| advance to blue van while avioding standing stones and other obstacles | 6 | 0 | 6 | 1196 |
| advance to blue van while avioding white pillars and other obstacles | 6 | 0 | 6 | 1060 |
| advance to covered car while avioding white pillars and other obstacles | 6 | 0 | 6 | 1116 |
| advance to girl in yellow shirt while avioding white pillars and other obstacles | 6 | 0 | 6 | 1158 |
| advance to green sphere while avioding tree clusters and other obstacles | 5 | 1 | 6 | 1083 |
| advance to man in black suit while avioding tree clusters and other obstacles | 5 | 1 | 6 | 1023 |
| advance to old television while avioding white pillars and other obstacles | 4 | 2 | 6 | 1195 |
| advance to police officer while avioding white pillars and other obstacles | 6 | 0 | 6 | 1162 |
| advance to red cube while avioding white pillars and other obstacles | 5 | 1 | 6 | 1210 |
| advance to wooden barrel while avioding standing stones and other obstacles | 5 | 1 | 6 | 1192 |
| go through and avoid the stacked boxes or other obstacles to reach the ambulance | 5 | 1 | 6 | 1092 |
| go through and avoid the stacked boxes or other obstacles to reach the round wooden table | 5 | 1 | 6 | 1082 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow wet floor sign | 5 | 1 | 6 | 1041 |
| go through and avoid the standing stones or other obstacles to reach the black leather sofa | 6 | 0 | 6 | 1061 |
| go through and avoid the stone field or other obstacles to reach the doctor in white coat | 5 | 1 | 6 | 1226 |
| go through and avoid the trees or other obstacles to reach the blue barrel | 5 | 1 | 6 | 1080 |
| go through and avoid the trees or other obstacles to reach the red wooden bench | 6 | 0 | 6 | 1102 |
| go through and avoid the white pillars or other obstacles to reach the green armchair | 6 | 0 | 6 | 1262 |
| go through and avoid the white pillars or other obstacles to reach the man in black suit | 5 | 1 | 6 | 1104 |
| go through and avoid the white pillars or other obstacles to reach the man in blue jacket | 4 | 2 | 6 | 1221 |
| go through and avoid the white pillars or other obstacles to reach the man in red shirt | 6 | 0 | 6 | 1144 |
| go through and avoid the white pillars or other obstacles to reach the orange cone | 5 | 1 | 6 | 1142 |
| go through and avoid the white pillars or other obstacles to reach the red wooden bench | 6 | 0 | 6 | 1188 |
| go through and avoid the white pillars or other obstacles to reach the yellow taxi | 5 | 1 | 6 | 1251 |
| advance to doctor in white coat while avioding white pillars and other obstacles | 5 | 0 | 5 | 919 |
| advance to green cube while avioding white pillars and other obstacles | 5 | 0 | 5 | 949 |
| advance to green hatchback while avioding white pillars and other obstacles | 5 | 0 | 5 | 968 |
| advance to grey armchair while avioding stone field and other obstacles | 5 | 0 | 5 | 869 |
| advance to grey armchair while avioding white pillars and other obstacles | 5 | 0 | 5 | 1108 |
| advance to man in blue jacket while avioding rocks and other obstacles | 4 | 1 | 5 | 930 |
| advance to man in brown jacket while avioding standing stones and other obstacles | 5 | 0 | 5 | 843 |
| advance to orange cylinder while avioding stone field and other obstacles | 4 | 1 | 5 | 861 |
| advance to police car while avioding white pillars and other obstacles | 4 | 1 | 5 | 883 |
| advance to red barrel while avioding trees and other obstacles | 5 | 0 | 5 | 933 |
| advance to red cube while avioding coloured poles and other obstacles | 4 | 1 | 5 | 902 |
| advance to red cube while avioding rocks and other obstacles | 5 | 0 | 5 | 1054 |
| advance to red fire truck while avioding white pillars and other obstacles | 4 | 1 | 5 | 888 |
| advance to red sedan while avioding white pillars and other obstacles | 4 | 1 | 5 | 993 |
| advance to red tool chest while avioding tree clusters and other obstacles | 4 | 1 | 5 | 903 |
| advance to white cabinet while avioding standing stones and other obstacles | 5 | 0 | 5 | 983 |
| advance to white cabinet while avioding tree clusters and other obstacles | 4 | 1 | 5 | 906 |
| advance to white plastic chair while avioding trees and other obstacles | 4 | 1 | 5 | 860 |
| advance to yellow sphere while avioding white pillars and other obstacles | 5 | 0 | 5 | 1004 |
| advance to yellow wet floor sign while avioding rocks and other obstacles | 5 | 0 | 5 | 899 |
| advance to yellow wet floor sign while avioding stacked boxes and other obstacles | 5 | 0 | 5 | 937 |
| go through and avoid the coloured poles or other obstacles to reach the yellow cube | 5 | 0 | 5 | 959 |
| go through and avoid the rocks or other obstacles to reach the blue cylinder | 4 | 1 | 5 | 961 |
| go through and avoid the rocks or other obstacles to reach the girl in yellow shirt | 4 | 1 | 5 | 884 |
| go through and avoid the rocks or other obstacles to reach the white cabinet | 5 | 0 | 5 | 1047 |
| go through and avoid the stacked boxes or other obstacles to reach the blue school chair | 5 | 0 | 5 | 919 |
| go through and avoid the stacked boxes or other obstacles to reach the red cylinder | 4 | 1 | 5 | 905 |
| go through and avoid the standing stones or other obstacles to reach the doctor in white coat | 5 | 0 | 5 | 972 |
| go through and avoid the stone field or other obstacles to reach the grey armchair | 4 | 1 | 5 | 925 |
| go through and avoid the tree clusters or other obstacles to reach the man in red shirt | 5 | 0 | 5 | 916 |
| go through and avoid the trees or other obstacles to reach the black leather sofa | 4 | 1 | 5 | 925 |
| go through and avoid the trees or other obstacles to reach the green sphere | 4 | 1 | 5 | 1022 |
| go through and avoid the white pillars or other obstacles to reach the beige sofa | 5 | 0 | 5 | 976 |
| go through and avoid the white pillars or other obstacles to reach the green sphere | 5 | 0 | 5 | 941 |
| go through and avoid the white pillars or other obstacles to reach the person in purple shirt | 4 | 1 | 5 | 898 |
| go through and avoid the white pillars or other obstacles to reach the red cube | 5 | 0 | 5 | 889 |
| go through and avoid the white pillars or other obstacles to reach the yellow cube | 5 | 0 | 5 | 929 |
| advance to bald man in green shirt while avioding coloured poles and other obstacles | 4 | 0 | 4 | 730 |
| advance to bald man in green shirt while avioding white pillars and other obstacles | 4 | 0 | 4 | 849 |
| advance to black leather sofa while avioding trees and other obstacles | 4 | 0 | 4 | 742 |
| advance to black leather sofa while avioding white pillars and other obstacles | 4 | 0 | 4 | 825 |
| advance to blue barrel while avioding stone field and other obstacles | 3 | 1 | 4 | 772 |
| advance to blue cone while avioding white pillars and other obstacles | 4 | 0 | 4 | 747 |
| advance to blue cylinder while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 716 |
| advance to blue cylinder while avioding tree clusters and other obstacles | 4 | 0 | 4 | 772 |
| advance to blue school chair while avioding tree clusters and other obstacles | 3 | 1 | 4 | 711 |
| advance to bronze whale statue while avioding coloured poles and other obstacles | 4 | 0 | 4 | 765 |
| advance to doctor in white coat while avioding trees and other obstacles | 4 | 0 | 4 | 772 |
| advance to green armchair while avioding coloured poles and other obstacles | 4 | 0 | 4 | 732 |
| advance to green armchair while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 666 |
| advance to green armchair while avioding standing stones and other obstacles | 4 | 0 | 4 | 824 |
| advance to green armchair while avioding stone field and other obstacles | 4 | 0 | 4 | 706 |
| advance to green armchair while avioding white pillars and other obstacles | 4 | 0 | 4 | 880 |
| advance to green garbage truck while avioding coloured poles and other obstacles | 3 | 1 | 4 | 668 |
| advance to green garbage truck while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 760 |
| advance to green hatchback while avioding coloured poles and other obstacles | 4 | 0 | 4 | 707 |
| advance to green sphere while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 725 |
| advance to green suitcase while avioding white pillars and other obstacles | 4 | 0 | 4 | 825 |
| advance to grey armchair while avioding trees and other obstacles | 4 | 0 | 4 | 743 |
| advance to man in blue jacket while avioding tree clusters and other obstacles | 4 | 0 | 4 | 713 |
| advance to man in brown jacket while avioding trees and other obstacles | 3 | 1 | 4 | 777 |
| advance to man in red shirt while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 747 |
| advance to old television while avioding coloured poles and other obstacles | 3 | 1 | 4 | 708 |
| advance to old television while avioding rocks and other obstacles | 4 | 0 | 4 | 690 |
| advance to orange cylinder while avioding rocks and other obstacles | 3 | 1 | 4 | 736 |
| advance to orange cylinder while avioding stacked boxes and other obstacles | 3 | 1 | 4 | 748 |
| advance to orange cylinder while avioding standing stones and other obstacles | 4 | 0 | 4 | 688 |
| advance to red barrel while avioding standing stones and other obstacles | 4 | 0 | 4 | 703 |
| advance to red cylinder while avioding tree clusters and other obstacles | 4 | 0 | 4 | 823 |
| advance to red fire extinguisher while avioding standing stones and other obstacles | 4 | 0 | 4 | 802 |
| advance to red fire extinguisher while avioding stone field and other obstacles | 3 | 1 | 4 | 712 |
| advance to red fire extinguisher while avioding tree clusters and other obstacles | 4 | 0 | 4 | 737 |
| advance to red fire extinguisher while avioding white pillars and other obstacles | 4 | 0 | 4 | 827 |
| advance to red sedan while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 736 |
| advance to red sedan while avioding stone field and other obstacles | 4 | 0 | 4 | 764 |
| advance to red tool chest while avioding coloured poles and other obstacles | 4 | 0 | 4 | 721 |
| advance to red tool chest while avioding white pillars and other obstacles | 4 | 0 | 4 | 908 |
| advance to round wooden table while avioding stacked boxes and other obstacles | 4 | 0 | 4 | 664 |
| advance to white cabinet while avioding coloured poles and other obstacles | 3 | 1 | 4 | 773 |
| advance to white cabinet while avioding rocks and other obstacles | 3 | 1 | 4 | 790 |
| advance to white cabinet while avioding white pillars and other obstacles | 4 | 0 | 4 | 752 |
| advance to wooden barrel while avioding trees and other obstacles | 3 | 1 | 4 | 645 |
| advance to yellow cube while avioding stone field and other obstacles | 4 | 0 | 4 | 792 |
| advance to yellow sphere while avioding coloured poles and other obstacles | 3 | 1 | 4 | 811 |
| advance to yellow suv while avioding coloured poles and other obstacles | 3 | 1 | 4 | 704 |
| advance to yellow wet floor sign while avioding standing stones and other obstacles | 4 | 0 | 4 | 719 |
| advance to yellow wet floor sign while avioding stone field and other obstacles | 4 | 0 | 4 | 743 |
| advance to yellow wet floor sign while avioding trees and other obstacles | 4 | 0 | 4 | 655 |
| go through and avoid the coloured poles or other obstacles to reach the bald man in green shirt | 3 | 1 | 4 | 709 |
| go through and avoid the coloured poles or other obstacles to reach the blue van | 4 | 0 | 4 | 775 |
| go through and avoid the coloured poles or other obstacles to reach the doctor in white coat | 3 | 1 | 4 | 753 |
| go through and avoid the coloured poles or other obstacles to reach the green garbage truck | 4 | 0 | 4 | 703 |
| go through and avoid the coloured poles or other obstacles to reach the green sphere | 4 | 0 | 4 | 675 |
| go through and avoid the coloured poles or other obstacles to reach the police car | 4 | 0 | 4 | 726 |
| go through and avoid the coloured poles or other obstacles to reach the red sedan | 4 | 0 | 4 | 761 |
| go through and avoid the coloured poles or other obstacles to reach the yellow generator | 4 | 0 | 4 | 697 |
| go through and avoid the coloured poles or other obstacles to reach the yellow suv | 4 | 0 | 4 | 742 |
| go through and avoid the rocks or other obstacles to reach the beige sofa | 4 | 0 | 4 | 766 |
| go through and avoid the rocks or other obstacles to reach the bronze whale statue | 3 | 1 | 4 | 707 |
| go through and avoid the rocks or other obstacles to reach the man in red shirt | 4 | 0 | 4 | 826 |
| go through and avoid the rocks or other obstacles to reach the orange cone | 4 | 0 | 4 | 724 |
| go through and avoid the rocks or other obstacles to reach the yellow cylinder | 4 | 0 | 4 | 695 |
| go through and avoid the rocks or other obstacles to reach the yellow wet floor sign | 3 | 1 | 4 | 843 |
| go through and avoid the stacked boxes or other obstacles to reach the beige sofa | 3 | 1 | 4 | 719 |
| go through and avoid the stacked boxes or other obstacles to reach the blue cone | 4 | 0 | 4 | 678 |
| go through and avoid the stacked boxes or other obstacles to reach the doctor in white coat | 4 | 0 | 4 | 699 |
| go through and avoid the stacked boxes or other obstacles to reach the white pickup truck | 4 | 0 | 4 | 734 |
| go through and avoid the standing stones or other obstacles to reach the bronze whale statue | 3 | 1 | 4 | 691 |
| go through and avoid the standing stones or other obstacles to reach the green armchair | 3 | 1 | 4 | 772 |
| go through and avoid the standing stones or other obstacles to reach the orange cone | 4 | 0 | 4 | 683 |
| go through and avoid the standing stones or other obstacles to reach the person in purple shirt | 4 | 0 | 4 | 699 |
| go through and avoid the standing stones or other obstacles to reach the red sedan | 4 | 0 | 4 | 729 |
| go through and avoid the stone field or other obstacles to reach the green hatchback | 4 | 0 | 4 | 754 |
| go through and avoid the stone field or other obstacles to reach the orange cylinder | 4 | 0 | 4 | 889 |
| go through and avoid the stone field or other obstacles to reach the red fire extinguisher | 4 | 0 | 4 | 731 |
| go through and avoid the stone field or other obstacles to reach the red tool chest | 4 | 0 | 4 | 732 |
| go through and avoid the tree clusters or other obstacles to reach the green armchair | 4 | 0 | 4 | 678 |
| go through and avoid the tree clusters or other obstacles to reach the green cube | 4 | 0 | 4 | 704 |
| go through and avoid the tree clusters or other obstacles to reach the orange cylinder | 3 | 1 | 4 | 695 |
| go through and avoid the tree clusters or other obstacles to reach the police officer | 4 | 0 | 4 | 759 |
| go through and avoid the tree clusters or other obstacles to reach the red barrel | 4 | 0 | 4 | 728 |
| go through and avoid the tree clusters or other obstacles to reach the white pickup truck | 3 | 1 | 4 | 706 |
| go through and avoid the tree clusters or other obstacles to reach the yellow cylinder | 3 | 1 | 4 | 677 |
| go through and avoid the tree clusters or other obstacles to reach the yellow suv | 4 | 0 | 4 | 836 |
| go through and avoid the trees or other obstacles to reach the blue cylinder | 3 | 1 | 4 | 763 |
| go through and avoid the trees or other obstacles to reach the girl in yellow shirt | 3 | 1 | 4 | 774 |
| go through and avoid the trees or other obstacles to reach the grey armchair | 3 | 1 | 4 | 662 |
| go through and avoid the trees or other obstacles to reach the red fire extinguisher | 4 | 0 | 4 | 681 |
| go through and avoid the trees or other obstacles to reach the wooden barrel | 4 | 0 | 4 | 694 |
| go through and avoid the white pillars or other obstacles to reach the bald man in green shirt | 3 | 1 | 4 | 744 |
| go through and avoid the white pillars or other obstacles to reach the doctor in white coat | 4 | 0 | 4 | 741 |
| go through and avoid the white pillars or other obstacles to reach the girl in yellow shirt | 4 | 0 | 4 | 914 |
| go through and avoid the white pillars or other obstacles to reach the green hatchback | 4 | 0 | 4 | 658 |
| go through and avoid the white pillars or other obstacles to reach the grey armchair | 4 | 0 | 4 | 753 |
| go through and avoid the white pillars or other obstacles to reach the red tool chest | 4 | 0 | 4 | 781 |
| go through and avoid the white pillars or other obstacles to reach the white pickup truck | 4 | 0 | 4 | 680 |
| go through and avoid the white pillars or other obstacles to reach the yellow generator | 4 | 0 | 4 | 825 |
| advance to ambulance while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 530 |
| advance to bald man in green shirt while avioding stone field and other obstacles | 3 | 0 | 3 | 550 |
| advance to beige sofa while avioding coloured poles and other obstacles | 3 | 0 | 3 | 554 |
| advance to beige sofa while avioding rocks and other obstacles | 2 | 1 | 3 | 601 |
| advance to black leather sofa while avioding stone field and other obstacles | 3 | 0 | 3 | 541 |
| advance to blue barrel while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 510 |
| advance to blue cone while avioding standing stones and other obstacles | 3 | 0 | 3 | 564 |
| advance to blue cone while avioding trees and other obstacles | 3 | 0 | 3 | 566 |
| advance to blue cylinder while avioding trees and other obstacles | 3 | 0 | 3 | 523 |
| advance to blue school chair while avioding stacked boxes and other obstacles | 2 | 1 | 3 | 514 |
| advance to blue school chair while avioding trees and other obstacles | 3 | 0 | 3 | 582 |
| advance to blue van while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 530 |
| advance to bronze whale statue while avioding rocks and other obstacles | 3 | 0 | 3 | 667 |
| advance to bronze whale statue while avioding standing stones and other obstacles | 3 | 0 | 3 | 563 |
| advance to bronze whale statue while avioding stone field and other obstacles | 3 | 0 | 3 | 540 |
| advance to covered car while avioding stone field and other obstacles | 2 | 1 | 3 | 485 |
| advance to doctor in white coat while avioding coloured poles and other obstacles | 3 | 0 | 3 | 534 |
| advance to girl in yellow shirt while avioding coloured poles and other obstacles | 3 | 0 | 3 | 602 |
| advance to girl in yellow shirt while avioding standing stones and other obstacles | 3 | 0 | 3 | 545 |
| advance to girl in yellow shirt while avioding trees and other obstacles | 3 | 0 | 3 | 549 |
| advance to green cube while avioding coloured poles and other obstacles | 3 | 0 | 3 | 577 |
| advance to green garbage truck while avioding white pillars and other obstacles | 3 | 0 | 3 | 607 |
| advance to green sphere while avioding trees and other obstacles | 3 | 0 | 3 | 533 |
| advance to green suitcase while avioding standing stones and other obstacles | 3 | 0 | 3 | 586 |
| advance to green suitcase while avioding tree clusters and other obstacles | 3 | 0 | 3 | 642 |
| advance to green suitcase while avioding trees and other obstacles | 3 | 0 | 3 | 591 |
| advance to man in black suit while avioding coloured poles and other obstacles | 3 | 0 | 3 | 534 |
| advance to man in black suit while avioding rocks and other obstacles | 3 | 0 | 3 | 599 |
| advance to man in black suit while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 495 |
| advance to man in black suit while avioding trees and other obstacles | 3 | 0 | 3 | 592 |
| advance to man in brown jacket while avioding rocks and other obstacles | 3 | 0 | 3 | 606 |
| advance to man in brown jacket while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 545 |
| advance to man in brown jacket while avioding stone field and other obstacles | 2 | 1 | 3 | 527 |
| advance to man in brown jacket while avioding tree clusters and other obstacles | 3 | 0 | 3 | 543 |
| advance to old television while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 514 |
| advance to orange cone while avioding coloured poles and other obstacles | 3 | 0 | 3 | 549 |
| advance to orange cone while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 529 |
| advance to orange cone while avioding tree clusters and other obstacles | 2 | 1 | 3 | 515 |
| advance to orange cone while avioding trees and other obstacles | 3 | 0 | 3 | 542 |
| advance to person in purple shirt while avioding rocks and other obstacles | 3 | 0 | 3 | 560 |
| advance to person in purple shirt while avioding stacked boxes and other obstacles | 2 | 1 | 3 | 483 |
| advance to police car while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 502 |
| advance to police car while avioding tree clusters and other obstacles | 3 | 0 | 3 | 521 |
| advance to police officer while avioding coloured poles and other obstacles | 3 | 0 | 3 | 519 |
| advance to police officer while avioding stone field and other obstacles | 3 | 0 | 3 | 604 |
| advance to police officer while avioding trees and other obstacles | 3 | 0 | 3 | 576 |
| advance to red barrel while avioding rocks and other obstacles | 3 | 0 | 3 | 596 |
| advance to red sedan while avioding rocks and other obstacles | 3 | 0 | 3 | 718 |
| advance to red tool chest while avioding trees and other obstacles | 3 | 0 | 3 | 506 |
| advance to red wooden bench while avioding coloured poles and other obstacles | 3 | 0 | 3 | 506 |
| advance to red wooden bench while avioding standing stones and other obstacles | 3 | 0 | 3 | 555 |
| advance to red wooden bench while avioding tree clusters and other obstacles | 3 | 0 | 3 | 511 |
| advance to round wooden table while avioding standing stones and other obstacles | 3 | 0 | 3 | 589 |
| advance to round wooden table while avioding trees and other obstacles | 3 | 0 | 3 | 497 |
| advance to round wooden table while avioding white pillars and other obstacles | 3 | 0 | 3 | 576 |
| advance to white cabinet while avioding stone field and other obstacles | 3 | 0 | 3 | 488 |
| advance to white pickup truck while avioding rocks and other obstacles | 3 | 0 | 3 | 709 |
| advance to yellow cube while avioding standing stones and other obstacles | 3 | 0 | 3 | 597 |
| advance to yellow cylinder while avioding stacked boxes and other obstacles | 3 | 0 | 3 | 559 |
| advance to yellow cylinder while avioding tree clusters and other obstacles | 3 | 0 | 3 | 631 |
| advance to yellow generator while avioding stone field and other obstacles | 3 | 0 | 3 | 584 |
| advance to yellow generator while avioding tree clusters and other obstacles | 3 | 0 | 3 | 590 |
| advance to yellow suv while avioding rocks and other obstacles | 3 | 0 | 3 | 630 |
| advance to yellow wet floor sign while avioding coloured poles and other obstacles | 3 | 0 | 3 | 566 |
| advance to yellow wet floor sign while avioding tree clusters and other obstacles | 3 | 0 | 3 | 545 |
| go through and avoid the coloured poles or other obstacles to reach the ambulance | 3 | 0 | 3 | 512 |
| go through and avoid the coloured poles or other obstacles to reach the blue school chair | 3 | 0 | 3 | 565 |
| go through and avoid the coloured poles or other obstacles to reach the man in black suit | 3 | 0 | 3 | 539 |
| go through and avoid the coloured poles or other obstacles to reach the old television | 3 | 0 | 3 | 572 |
| go through and avoid the coloured poles or other obstacles to reach the orange cone | 2 | 1 | 3 | 529 |
| go through and avoid the coloured poles or other obstacles to reach the red cube | 3 | 0 | 3 | 564 |
| go through and avoid the coloured poles or other obstacles to reach the red tool chest | 2 | 1 | 3 | 526 |
| go through and avoid the coloured poles or other obstacles to reach the round wooden table | 3 | 0 | 3 | 554 |
| go through and avoid the coloured poles or other obstacles to reach the yellow wet floor sign | 3 | 0 | 3 | 540 |
| go through and avoid the rocks or other obstacles to reach the bald man in green shirt | 3 | 0 | 3 | 526 |
| go through and avoid the rocks or other obstacles to reach the black leather sofa | 3 | 0 | 3 | 684 |
| go through and avoid the rocks or other obstacles to reach the blue cone | 3 | 0 | 3 | 678 |
| go through and avoid the rocks or other obstacles to reach the covered car | 3 | 0 | 3 | 490 |
| go through and avoid the rocks or other obstacles to reach the orange cylinder | 3 | 0 | 3 | 646 |
| go through and avoid the rocks or other obstacles to reach the person in purple shirt | 3 | 0 | 3 | 615 |
| go through and avoid the rocks or other obstacles to reach the red cylinder | 3 | 0 | 3 | 543 |
| go through and avoid the rocks or other obstacles to reach the red fire extinguisher | 3 | 0 | 3 | 587 |
| go through and avoid the rocks or other obstacles to reach the white pickup truck | 3 | 0 | 3 | 684 |
| go through and avoid the rocks or other obstacles to reach the yellow cube | 3 | 0 | 3 | 582 |
| go through and avoid the rocks or other obstacles to reach the yellow suv | 3 | 0 | 3 | 647 |
| go through and avoid the stacked boxes or other obstacles to reach the bald man in green shirt | 3 | 0 | 3 | 518 |
| go through and avoid the stacked boxes or other obstacles to reach the girl in yellow shirt | 3 | 0 | 3 | 542 |
| go through and avoid the stacked boxes or other obstacles to reach the green hatchback | 3 | 0 | 3 | 490 |
| go through and avoid the stacked boxes or other obstacles to reach the green sphere | 2 | 1 | 3 | 524 |
| go through and avoid the stacked boxes or other obstacles to reach the orange cylinder | 3 | 0 | 3 | 526 |
| go through and avoid the stacked boxes or other obstacles to reach the person in purple shirt | 3 | 0 | 3 | 492 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow suv | 3 | 0 | 3 | 516 |
| go through and avoid the standing stones or other obstacles to reach the blue barrel | 3 | 0 | 3 | 600 |
| go through and avoid the standing stones or other obstacles to reach the blue cone | 3 | 0 | 3 | 540 |
| go through and avoid the standing stones or other obstacles to reach the green suitcase | 3 | 0 | 3 | 647 |
| go through and avoid the standing stones or other obstacles to reach the man in black suit | 3 | 0 | 3 | 549 |
| go through and avoid the standing stones or other obstacles to reach the man in brown jacket | 2 | 1 | 3 | 576 |
| go through and avoid the standing stones or other obstacles to reach the red cylinder | 3 | 0 | 3 | 613 |
| go through and avoid the standing stones or other obstacles to reach the white plastic chair | 3 | 0 | 3 | 673 |
| go through and avoid the standing stones or other obstacles to reach the yellow cube | 3 | 0 | 3 | 691 |
| go through and avoid the standing stones or other obstacles to reach the yellow wet floor sign | 2 | 1 | 3 | 512 |
| go through and avoid the stone field or other obstacles to reach the bald man in green shirt | 2 | 1 | 3 | 562 |
| go through and avoid the stone field or other obstacles to reach the blue barrel | 3 | 0 | 3 | 548 |
| go through and avoid the stone field or other obstacles to reach the bronze whale statue | 3 | 0 | 3 | 541 |
| go through and avoid the stone field or other obstacles to reach the covered car | 3 | 0 | 3 | 569 |
| go through and avoid the stone field or other obstacles to reach the green suitcase | 3 | 0 | 3 | 523 |
| go through and avoid the stone field or other obstacles to reach the man in black suit | 3 | 0 | 3 | 606 |
| go through and avoid the stone field or other obstacles to reach the man in blue jacket | 3 | 0 | 3 | 540 |
| go through and avoid the stone field or other obstacles to reach the man in brown jacket | 3 | 0 | 3 | 590 |
| go through and avoid the stone field or other obstacles to reach the red cylinder | 3 | 0 | 3 | 540 |
| go through and avoid the stone field or other obstacles to reach the red wooden bench | 3 | 0 | 3 | 553 |
| go through and avoid the stone field or other obstacles to reach the yellow taxi | 3 | 0 | 3 | 625 |
| go through and avoid the tree clusters or other obstacles to reach the blue school chair | 3 | 0 | 3 | 525 |
| go through and avoid the tree clusters or other obstacles to reach the doctor in white coat | 3 | 0 | 3 | 605 |
| go through and avoid the tree clusters or other obstacles to reach the green suitcase | 3 | 0 | 3 | 567 |
| go through and avoid the tree clusters or other obstacles to reach the man in black suit | 3 | 0 | 3 | 656 |
| go through and avoid the tree clusters or other obstacles to reach the man in brown jacket | 3 | 0 | 3 | 506 |
| go through and avoid the tree clusters or other obstacles to reach the old television | 3 | 0 | 3 | 540 |
| go through and avoid the tree clusters or other obstacles to reach the orange cone | 3 | 0 | 3 | 587 |
| go through and avoid the tree clusters or other obstacles to reach the person in purple shirt | 3 | 0 | 3 | 508 |
| go through and avoid the tree clusters or other obstacles to reach the red cube | 3 | 0 | 3 | 513 |
| go through and avoid the tree clusters or other obstacles to reach the red sedan | 3 | 0 | 3 | 613 |
| go through and avoid the tree clusters or other obstacles to reach the round wooden table | 3 | 0 | 3 | 526 |
| go through and avoid the tree clusters or other obstacles to reach the white plastic chair | 3 | 0 | 3 | 582 |
| go through and avoid the trees or other obstacles to reach the bald man in green shirt | 3 | 0 | 3 | 555 |
| go through and avoid the trees or other obstacles to reach the beige sofa | 3 | 0 | 3 | 505 |
| go through and avoid the trees or other obstacles to reach the green suitcase | 3 | 0 | 3 | 501 |
| go through and avoid the trees or other obstacles to reach the man in blue jacket | 3 | 0 | 3 | 500 |
| go through and avoid the trees or other obstacles to reach the orange cone | 2 | 1 | 3 | 601 |
| go through and avoid the trees or other obstacles to reach the white plastic chair | 3 | 0 | 3 | 547 |
| go through and avoid the trees or other obstacles to reach the yellow cube | 3 | 0 | 3 | 585 |
| go through and avoid the white pillars or other obstacles to reach the blue cone | 3 | 0 | 3 | 571 |
| go through and avoid the white pillars or other obstacles to reach the blue school chair | 3 | 0 | 3 | 632 |
| go through and avoid the white pillars or other obstacles to reach the bronze whale statue | 3 | 0 | 3 | 554 |
| go through and avoid the white pillars or other obstacles to reach the green cube | 3 | 0 | 3 | 599 |
| go through and avoid the white pillars or other obstacles to reach the police officer | 3 | 0 | 3 | 555 |
| go through and avoid the white pillars or other obstacles to reach the red fire extinguisher | 3 | 0 | 3 | 636 |
| go through and avoid the white pillars or other obstacles to reach the red fire truck | 3 | 0 | 3 | 586 |
| go through and avoid the white pillars or other obstacles to reach the yellow sphere | 2 | 1 | 3 | 541 |
| go through and avoid the white pillars or other obstacles to reach the yellow wet floor sign | 3 | 0 | 3 | 596 |
| advance to ambulance while avioding stone field and other obstacles | 2 | 0 | 2 | 433 |
| advance to ambulance while avioding white pillars and other obstacles | 2 | 0 | 2 | 356 |
| advance to bald man in green shirt while avioding rocks and other obstacles | 2 | 0 | 2 | 332 |
| advance to bald man in green shirt while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 343 |
| advance to bald man in green shirt while avioding tree clusters and other obstacles | 2 | 0 | 2 | 473 |
| advance to beige sofa while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 321 |
| advance to beige sofa while avioding standing stones and other obstacles | 2 | 0 | 2 | 369 |
| advance to black leather sofa while avioding coloured poles and other obstacles | 2 | 0 | 2 | 331 |
| advance to black leather sofa while avioding rocks and other obstacles | 2 | 0 | 2 | 366 |
| advance to blue barrel while avioding rocks and other obstacles | 2 | 0 | 2 | 393 |
| advance to blue barrel while avioding standing stones and other obstacles | 2 | 0 | 2 | 439 |
| advance to blue cone while avioding coloured poles and other obstacles | 2 | 0 | 2 | 349 |
| advance to blue cone while avioding stone field and other obstacles | 2 | 0 | 2 | 337 |
| advance to blue cylinder while avioding rocks and other obstacles | 2 | 0 | 2 | 442 |
| advance to blue cylinder while avioding standing stones and other obstacles | 2 | 0 | 2 | 475 |
| advance to blue school chair while avioding rocks and other obstacles | 2 | 0 | 2 | 352 |
| advance to blue van while avioding coloured poles and other obstacles | 1 | 1 | 2 | 355 |
| advance to bronze whale statue while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 398 |
| advance to bronze whale statue while avioding trees and other obstacles | 2 | 0 | 2 | 332 |
| advance to bronze whale statue while avioding white pillars and other obstacles | 2 | 0 | 2 | 510 |
| advance to covered car while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 320 |
| advance to covered car while avioding standing stones and other obstacles | 2 | 0 | 2 | 413 |
| advance to covered car while avioding tree clusters and other obstacles | 2 | 0 | 2 | 376 |
| advance to doctor in white coat while avioding standing stones and other obstacles | 1 | 1 | 2 | 373 |
| advance to girl in yellow shirt while avioding rocks and other obstacles | 2 | 0 | 2 | 414 |
| advance to girl in yellow shirt while avioding stone field and other obstacles | 2 | 0 | 2 | 359 |
| advance to girl in yellow shirt while avioding tree clusters and other obstacles | 2 | 0 | 2 | 419 |
| advance to green cube while avioding rocks and other obstacles | 2 | 0 | 2 | 384 |
| advance to green hatchback while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 335 |
| advance to green hatchback while avioding standing stones and other obstacles | 2 | 0 | 2 | 406 |
| advance to green suitcase while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 365 |
| advance to green suitcase while avioding stone field and other obstacles | 2 | 0 | 2 | 342 |
| advance to grey armchair while avioding coloured poles and other obstacles | 2 | 0 | 2 | 395 |
| advance to man in black suit while avioding stone field and other obstacles | 2 | 0 | 2 | 374 |
| advance to man in blue jacket while avioding standing stones and other obstacles | 2 | 0 | 2 | 335 |
| advance to man in red shirt while avioding white pillars and other obstacles | 2 | 0 | 2 | 358 |
| advance to old television while avioding trees and other obstacles | 2 | 0 | 2 | 342 |
| advance to orange cone while avioding standing stones and other obstacles | 2 | 0 | 2 | 489 |
| advance to orange cylinder while avioding tree clusters and other obstacles | 2 | 0 | 2 | 381 |
| advance to orange cylinder while avioding trees and other obstacles | 2 | 0 | 2 | 387 |
| advance to person in purple shirt while avioding stone field and other obstacles | 2 | 0 | 2 | 456 |
| advance to person in purple shirt while avioding tree clusters and other obstacles | 2 | 0 | 2 | 397 |
| advance to person in purple shirt while avioding trees and other obstacles | 2 | 0 | 2 | 343 |
| advance to police car while avioding trees and other obstacles | 2 | 0 | 2 | 397 |
| advance to police officer while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 337 |
| advance to red cube while avioding standing stones and other obstacles | 2 | 0 | 2 | 376 |
| advance to red cube while avioding stone field and other obstacles | 2 | 0 | 2 | 459 |
| advance to red cube while avioding trees and other obstacles | 2 | 0 | 2 | 369 |
| advance to red cylinder while avioding rocks and other obstacles | 2 | 0 | 2 | 392 |
| advance to red cylinder while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 320 |
| advance to red fire extinguisher while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 354 |
| advance to red fire extinguisher while avioding trees and other obstacles | 2 | 0 | 2 | 427 |
| advance to red fire truck while avioding coloured poles and other obstacles | 2 | 0 | 2 | 353 |
| advance to red fire truck while avioding standing stones and other obstacles | 2 | 0 | 2 | 423 |
| advance to red sedan while avioding coloured poles and other obstacles | 2 | 0 | 2 | 343 |
| advance to red tool chest while avioding rocks and other obstacles | 2 | 0 | 2 | 341 |
| advance to red wooden bench while avioding stacked boxes and other obstacles | 2 | 0 | 2 | 369 |
| advance to red wooden bench while avioding stone field and other obstacles | 2 | 0 | 2 | 392 |
| advance to red wooden bench while avioding white pillars and other obstacles | 2 | 0 | 2 | 383 |
| advance to round wooden table while avioding coloured poles and other obstacles | 2 | 0 | 2 | 341 |
| advance to white cabinet while avioding trees and other obstacles | 2 | 0 | 2 | 384 |
| advance to white pickup truck while avioding tree clusters and other obstacles | 2 | 0 | 2 | 393 |
| advance to white pickup truck while avioding white pillars and other obstacles | 2 | 0 | 2 | 370 |
| advance to white plastic chair while avioding coloured poles and other obstacles | 2 | 0 | 2 | 412 |
| advance to white plastic chair while avioding tree clusters and other obstacles | 2 | 0 | 2 | 345 |
| advance to wooden barrel while avioding rocks and other obstacles | 2 | 0 | 2 | 411 |
| advance to wooden barrel while avioding stone field and other obstacles | 2 | 0 | 2 | 346 |
| advance to wooden barrel while avioding tree clusters and other obstacles | 2 | 0 | 2 | 344 |
| advance to yellow cube while avioding coloured poles and other obstacles | 1 | 1 | 2 | 383 |
| advance to yellow cube while avioding rocks and other obstacles | 2 | 0 | 2 | 354 |
| advance to yellow cube while avioding tree clusters and other obstacles | 2 | 0 | 2 | 346 |
| advance to yellow cube while avioding trees and other obstacles | 2 | 0 | 2 | 361 |
| advance to yellow cylinder while avioding stone field and other obstacles | 2 | 0 | 2 | 362 |
| advance to yellow sphere while avioding tree clusters and other obstacles | 2 | 0 | 2 | 337 |
| advance to yellow sphere while avioding trees and other obstacles | 2 | 0 | 2 | 423 |
| advance to yellow wet floor sign while avioding white pillars and other obstacles | 2 | 0 | 2 | 366 |
| go through and avoid the coloured poles or other obstacles to reach the blue cone | 2 | 0 | 2 | 358 |
| go through and avoid the coloured poles or other obstacles to reach the covered car | 2 | 0 | 2 | 337 |
| go through and avoid the coloured poles or other obstacles to reach the girl in yellow shirt | 2 | 0 | 2 | 353 |
| go through and avoid the coloured poles or other obstacles to reach the green armchair | 2 | 0 | 2 | 359 |
| go through and avoid the coloured poles or other obstacles to reach the green hatchback | 2 | 0 | 2 | 336 |
| go through and avoid the coloured poles or other obstacles to reach the man in brown jacket | 2 | 0 | 2 | 355 |
| go through and avoid the coloured poles or other obstacles to reach the man in red shirt | 2 | 0 | 2 | 359 |
| go through and avoid the coloured poles or other obstacles to reach the police officer | 2 | 0 | 2 | 414 |
| go through and avoid the coloured poles or other obstacles to reach the white cabinet | 2 | 0 | 2 | 405 |
| go through and avoid the coloured poles or other obstacles to reach the white plastic chair | 2 | 0 | 2 | 395 |
| go through and avoid the coloured poles or other obstacles to reach the yellow sphere | 2 | 0 | 2 | 328 |
| go through and avoid the rocks or other obstacles to reach the blue barrel | 2 | 0 | 2 | 398 |
| go through and avoid the rocks or other obstacles to reach the blue school chair | 2 | 0 | 2 | 372 |
| go through and avoid the rocks or other obstacles to reach the blue van | 2 | 0 | 2 | 398 |
| go through and avoid the rocks or other obstacles to reach the green armchair | 2 | 0 | 2 | 384 |
| go through and avoid the rocks or other obstacles to reach the green garbage truck | 2 | 0 | 2 | 439 |
| go through and avoid the rocks or other obstacles to reach the green hatchback | 2 | 0 | 2 | 429 |
| go through and avoid the rocks or other obstacles to reach the man in blue jacket | 2 | 0 | 2 | 341 |
| go through and avoid the rocks or other obstacles to reach the man in brown jacket | 2 | 0 | 2 | 461 |
| go through and avoid the rocks or other obstacles to reach the old television | 2 | 0 | 2 | 341 |
| go through and avoid the rocks or other obstacles to reach the police officer | 2 | 0 | 2 | 418 |
| go through and avoid the rocks or other obstacles to reach the red barrel | 2 | 0 | 2 | 411 |
| go through and avoid the rocks or other obstacles to reach the red tool chest | 2 | 0 | 2 | 344 |
| go through and avoid the rocks or other obstacles to reach the white plastic chair | 2 | 0 | 2 | 337 |
| go through and avoid the rocks or other obstacles to reach the wooden barrel | 2 | 0 | 2 | 408 |
| go through and avoid the rocks or other obstacles to reach the yellow generator | 2 | 0 | 2 | 332 |
| go through and avoid the stacked boxes or other obstacles to reach the blue cylinder | 2 | 0 | 2 | 328 |
| go through and avoid the stacked boxes or other obstacles to reach the bronze whale statue | 2 | 0 | 2 | 321 |
| go through and avoid the stacked boxes or other obstacles to reach the covered car | 2 | 0 | 2 | 328 |
| go through and avoid the stacked boxes or other obstacles to reach the green suitcase | 2 | 0 | 2 | 391 |
| go through and avoid the stacked boxes or other obstacles to reach the grey armchair | 2 | 0 | 2 | 363 |
| go through and avoid the stacked boxes or other obstacles to reach the man in black suit | 2 | 0 | 2 | 342 |
| go through and avoid the stacked boxes or other obstacles to reach the old television | 2 | 0 | 2 | 320 |
| go through and avoid the stacked boxes or other obstacles to reach the orange cone | 2 | 0 | 2 | 358 |
| go through and avoid the stacked boxes or other obstacles to reach the police officer | 2 | 0 | 2 | 337 |
| go through and avoid the stacked boxes or other obstacles to reach the red fire extinguisher | 2 | 0 | 2 | 411 |
| go through and avoid the stacked boxes or other obstacles to reach the white plastic chair | 2 | 0 | 2 | 353 |
| go through and avoid the stacked boxes or other obstacles to reach the wooden barrel | 2 | 0 | 2 | 354 |
| go through and avoid the standing stones or other obstacles to reach the blue cylinder | 2 | 0 | 2 | 385 |
| go through and avoid the standing stones or other obstacles to reach the man in red shirt | 2 | 0 | 2 | 341 |
| go through and avoid the standing stones or other obstacles to reach the orange cylinder | 2 | 0 | 2 | 358 |
| go through and avoid the standing stones or other obstacles to reach the police officer | 2 | 0 | 2 | 426 |
| go through and avoid the standing stones or other obstacles to reach the red cube | 2 | 0 | 2 | 393 |
| go through and avoid the standing stones or other obstacles to reach the red fire extinguisher | 2 | 0 | 2 | 361 |
| go through and avoid the standing stones or other obstacles to reach the red fire truck | 2 | 0 | 2 | 466 |
| go through and avoid the standing stones or other obstacles to reach the red wooden bench | 2 | 0 | 2 | 407 |
| go through and avoid the standing stones or other obstacles to reach the yellow cylinder | 2 | 0 | 2 | 359 |
| go through and avoid the standing stones or other obstacles to reach the yellow generator | 2 | 0 | 2 | 362 |
| go through and avoid the stone field or other obstacles to reach the beige sofa | 2 | 0 | 2 | 384 |
| go through and avoid the stone field or other obstacles to reach the blue cone | 2 | 0 | 2 | 394 |
| go through and avoid the stone field or other obstacles to reach the blue cylinder | 2 | 0 | 2 | 388 |
| go through and avoid the stone field or other obstacles to reach the blue school chair | 2 | 0 | 2 | 392 |
| go through and avoid the stone field or other obstacles to reach the girl in yellow shirt | 2 | 0 | 2 | 378 |
| go through and avoid the stone field or other obstacles to reach the green armchair | 1 | 1 | 2 | 373 |
| go through and avoid the stone field or other obstacles to reach the green cube | 2 | 0 | 2 | 337 |
| go through and avoid the stone field or other obstacles to reach the man in red shirt | 2 | 0 | 2 | 378 |
| go through and avoid the stone field or other obstacles to reach the person in purple shirt | 2 | 0 | 2 | 391 |
| go through and avoid the stone field or other obstacles to reach the police officer | 2 | 0 | 2 | 459 |
| go through and avoid the stone field or other obstacles to reach the red cube | 2 | 0 | 2 | 391 |
| go through and avoid the stone field or other obstacles to reach the red sedan | 2 | 0 | 2 | 423 |
| go through and avoid the stone field or other obstacles to reach the round wooden table | 2 | 0 | 2 | 366 |
| go through and avoid the stone field or other obstacles to reach the white pickup truck | 2 | 0 | 2 | 429 |
| go through and avoid the stone field or other obstacles to reach the wooden barrel | 2 | 0 | 2 | 379 |
| go through and avoid the stone field or other obstacles to reach the yellow generator | 2 | 0 | 2 | 377 |
| go through and avoid the tree clusters or other obstacles to reach the bald man in green shirt | 2 | 0 | 2 | 368 |
| go through and avoid the tree clusters or other obstacles to reach the beige sofa | 2 | 0 | 2 | 341 |
| go through and avoid the tree clusters or other obstacles to reach the blue van | 2 | 0 | 2 | 426 |
| go through and avoid the tree clusters or other obstacles to reach the bronze whale statue | 2 | 0 | 2 | 407 |
| go through and avoid the tree clusters or other obstacles to reach the green hatchback | 2 | 0 | 2 | 335 |
| go through and avoid the tree clusters or other obstacles to reach the green sphere | 2 | 0 | 2 | 369 |
| go through and avoid the tree clusters or other obstacles to reach the red fire truck | 2 | 0 | 2 | 393 |
| go through and avoid the tree clusters or other obstacles to reach the yellow sphere | 2 | 0 | 2 | 331 |
| go through and avoid the trees or other obstacles to reach the doctor in white coat | 2 | 0 | 2 | 328 |
| go through and avoid the trees or other obstacles to reach the green armchair | 2 | 0 | 2 | 359 |
| go through and avoid the trees or other obstacles to reach the man in black suit | 2 | 0 | 2 | 350 |
| go through and avoid the trees or other obstacles to reach the man in brown jacket | 2 | 0 | 2 | 351 |
| go through and avoid the trees or other obstacles to reach the old television | 2 | 0 | 2 | 338 |
| go through and avoid the trees or other obstacles to reach the red barrel | 1 | 1 | 2 | 344 |
| go through and avoid the trees or other obstacles to reach the red cube | 2 | 0 | 2 | 341 |
| go through and avoid the trees or other obstacles to reach the red fire truck | 2 | 0 | 2 | 437 |
| go through and avoid the trees or other obstacles to reach the round wooden table | 2 | 0 | 2 | 402 |
| go through and avoid the trees or other obstacles to reach the yellow wet floor sign | 1 | 1 | 2 | 352 |
| go through and avoid the white pillars or other obstacles to reach the ambulance | 2 | 0 | 2 | 384 |
| go through and avoid the white pillars or other obstacles to reach the blue van | 2 | 0 | 2 | 447 |
| go through and avoid the white pillars or other obstacles to reach the green suitcase | 2 | 0 | 2 | 420 |
| go through and avoid the white pillars or other obstacles to reach the white plastic chair | 2 | 0 | 2 | 392 |
| go through and avoid the white pillars or other obstacles to reach the yellow suv | 2 | 0 | 2 | 383 |
| advance to ambulance while avioding coloured poles and other obstacles | 1 | 0 | 1 | 218 |
| advance to ambulance while avioding standing stones and other obstacles | 1 | 0 | 1 | 173 |
| advance to ambulance while avioding trees and other obstacles | 1 | 0 | 1 | 188 |
| advance to bald man in green shirt while avioding trees and other obstacles | 1 | 0 | 1 | 156 |
| advance to black leather sofa while avioding tree clusters and other obstacles | 1 | 0 | 1 | 227 |
| advance to blue barrel while avioding tree clusters and other obstacles | 1 | 0 | 1 | 211 |
| advance to blue barrel while avioding trees and other obstacles | 1 | 0 | 1 | 186 |
| advance to blue cone while avioding rocks and other obstacles | 1 | 0 | 1 | 201 |
| advance to blue cone while avioding tree clusters and other obstacles | 1 | 0 | 1 | 241 |
| advance to blue cylinder while avioding stone field and other obstacles | 1 | 0 | 1 | 203 |
| advance to blue school chair while avioding coloured poles and other obstacles | 1 | 0 | 1 | 185 |
| advance to blue school chair while avioding stone field and other obstacles | 1 | 0 | 1 | 184 |
| advance to blue van while avioding rocks and other obstacles | 1 | 0 | 1 | 171 |
| advance to blue van while avioding stone field and other obstacles | 1 | 0 | 1 | 208 |
| advance to blue van while avioding tree clusters and other obstacles | 1 | 0 | 1 | 184 |
| advance to bronze whale statue while avioding tree clusters and other obstacles | 1 | 0 | 1 | 190 |
| advance to covered car while avioding trees and other obstacles | 1 | 0 | 1 | 194 |
| advance to doctor in white coat while avioding rocks and other obstacles | 1 | 0 | 1 | 167 |
| advance to doctor in white coat while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 158 |
| advance to doctor in white coat while avioding stone field and other obstacles | 1 | 0 | 1 | 220 |
| advance to girl in yellow shirt while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 195 |
| advance to green armchair while avioding rocks and other obstacles | 1 | 0 | 1 | 160 |
| advance to green armchair while avioding trees and other obstacles | 1 | 0 | 1 | 156 |
| advance to green cube while avioding stone field and other obstacles | 1 | 0 | 1 | 200 |
| advance to green garbage truck while avioding rocks and other obstacles | 1 | 0 | 1 | 169 |
| advance to green garbage truck while avioding standing stones and other obstacles | 1 | 0 | 1 | 174 |
| advance to green garbage truck while avioding trees and other obstacles | 1 | 0 | 1 | 265 |
| advance to green hatchback while avioding stone field and other obstacles | 1 | 0 | 1 | 170 |
| advance to green sphere while avioding coloured poles and other obstacles | 1 | 0 | 1 | 181 |
| advance to green suitcase while avioding coloured poles and other obstacles | 1 | 0 | 1 | 183 |
| advance to grey armchair while avioding rocks and other obstacles | 1 | 0 | 1 | 195 |
| advance to grey armchair while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 166 |
| advance to grey armchair while avioding standing stones and other obstacles | 1 | 0 | 1 | 185 |
| advance to man in black suit while avioding standing stones and other obstacles | 1 | 0 | 1 | 189 |
| advance to man in blue jacket while avioding coloured poles and other obstacles | 1 | 0 | 1 | 189 |
| advance to man in blue jacket while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 163 |
| advance to man in brown jacket while avioding coloured poles and other obstacles | 1 | 0 | 1 | 163 |
| advance to man in brown jacket while avioding white pillars and other obstacles | 1 | 0 | 1 | 187 |
| advance to man in red shirt while avioding coloured poles and other obstacles | 1 | 0 | 1 | 170 |
| advance to man in red shirt while avioding rocks and other obstacles | 1 | 0 | 1 | 168 |
| advance to man in red shirt while avioding standing stones and other obstacles | 1 | 0 | 1 | 165 |
| advance to man in red shirt while avioding tree clusters and other obstacles | 1 | 0 | 1 | 158 |
| advance to man in red shirt while avioding trees and other obstacles | 1 | 0 | 1 | 172 |
| advance to old television while avioding standing stones and other obstacles | 1 | 0 | 1 | 188 |
| advance to old television while avioding stone field and other obstacles | 1 | 0 | 1 | 178 |
| advance to old television while avioding tree clusters and other obstacles | 1 | 0 | 1 | 162 |
| advance to person in purple shirt while avioding coloured poles and other obstacles | 1 | 0 | 1 | 200 |
| advance to person in purple shirt while avioding standing stones and other obstacles | 1 | 0 | 1 | 197 |
| advance to police car while avioding rocks and other obstacles | 1 | 0 | 1 | 218 |
| advance to police car while avioding stone field and other obstacles | 1 | 0 | 1 | 266 |
| advance to police officer while avioding rocks and other obstacles | 1 | 0 | 1 | 171 |
| advance to police officer while avioding tree clusters and other obstacles | 1 | 0 | 1 | 204 |
| advance to red barrel while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 165 |
| advance to red barrel while avioding tree clusters and other obstacles | 1 | 0 | 1 | 180 |
| advance to red cube while avioding tree clusters and other obstacles | 1 | 0 | 1 | 161 |
| advance to red cylinder while avioding standing stones and other obstacles | 1 | 0 | 1 | 170 |
| advance to red cylinder while avioding stone field and other obstacles | 1 | 0 | 1 | 167 |
| advance to red cylinder while avioding trees and other obstacles | 1 | 0 | 1 | 165 |
| advance to red fire extinguisher while avioding coloured poles and other obstacles | 1 | 0 | 1 | 206 |
| advance to red fire truck while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 155 |
| advance to red fire truck while avioding stone field and other obstacles | 1 | 0 | 1 | 194 |
| advance to red fire truck while avioding tree clusters and other obstacles | 1 | 0 | 1 | 171 |
| advance to red fire truck while avioding trees and other obstacles | 1 | 0 | 1 | 198 |
| advance to red tool chest while avioding standing stones and other obstacles | 1 | 0 | 1 | 188 |
| advance to red wooden bench while avioding rocks and other obstacles | 1 | 0 | 1 | 180 |
| advance to red wooden bench while avioding trees and other obstacles | 0 | 1 | 1 | 243 |
| advance to round wooden table while avioding rocks and other obstacles | 1 | 0 | 1 | 202 |
| advance to round wooden table while avioding stone field and other obstacles | 1 | 0 | 1 | 178 |
| advance to round wooden table while avioding tree clusters and other obstacles | 1 | 0 | 1 | 186 |
| advance to white pickup truck while avioding coloured poles and other obstacles | 1 | 0 | 1 | 168 |
| advance to white pickup truck while avioding trees and other obstacles | 1 | 0 | 1 | 177 |
| advance to white plastic chair while avioding rocks and other obstacles | 1 | 0 | 1 | 207 |
| advance to white plastic chair while avioding standing stones and other obstacles | 1 | 0 | 1 | 162 |
| advance to white plastic chair while avioding stone field and other obstacles | 1 | 0 | 1 | 208 |
| advance to wooden barrel while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 227 |
| advance to yellow cylinder while avioding rocks and other obstacles | 1 | 0 | 1 | 200 |
| advance to yellow cylinder while avioding standing stones and other obstacles | 1 | 0 | 1 | 166 |
| advance to yellow cylinder while avioding trees and other obstacles | 1 | 0 | 1 | 167 |
| advance to yellow generator while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 176 |
| advance to yellow generator while avioding standing stones and other obstacles | 1 | 0 | 1 | 167 |
| advance to yellow generator while avioding trees and other obstacles | 1 | 0 | 1 | 192 |
| advance to yellow sphere while avioding stacked boxes and other obstacles | 1 | 0 | 1 | 171 |
| advance to yellow suv while avioding stone field and other obstacles | 1 | 0 | 1 | 167 |
| advance to yellow suv while avioding tree clusters and other obstacles | 1 | 0 | 1 | 229 |
| advance to yellow suv while avioding trees and other obstacles | 1 | 0 | 1 | 175 |
| advance to yellow taxi while avioding coloured poles and other obstacles | 1 | 0 | 1 | 223 |
| advance to yellow taxi while avioding rocks and other obstacles | 1 | 0 | 1 | 159 |
| advance to yellow taxi while avioding standing stones and other obstacles | 1 | 0 | 1 | 214 |
| go through and avoid the coloured poles or other obstacles to reach the beige sofa | 1 | 0 | 1 | 167 |
| go through and avoid the coloured poles or other obstacles to reach the bronze whale statue | 1 | 0 | 1 | 161 |
| go through and avoid the coloured poles or other obstacles to reach the green cube | 1 | 0 | 1 | 178 |
| go through and avoid the coloured poles or other obstacles to reach the man in blue jacket | 1 | 0 | 1 | 171 |
| go through and avoid the coloured poles or other obstacles to reach the person in purple shirt | 1 | 0 | 1 | 167 |
| go through and avoid the coloured poles or other obstacles to reach the red fire extinguisher | 1 | 0 | 1 | 216 |
| go through and avoid the coloured poles or other obstacles to reach the red fire truck | 1 | 0 | 1 | 231 |
| go through and avoid the coloured poles or other obstacles to reach the red wooden bench | 1 | 0 | 1 | 166 |
| go through and avoid the coloured poles or other obstacles to reach the yellow taxi | 1 | 0 | 1 | 187 |
| go through and avoid the rocks or other obstacles to reach the ambulance | 1 | 0 | 1 | 207 |
| go through and avoid the rocks or other obstacles to reach the green cube | 1 | 0 | 1 | 194 |
| go through and avoid the rocks or other obstacles to reach the grey armchair | 1 | 0 | 1 | 206 |
| go through and avoid the rocks or other obstacles to reach the man in black suit | 1 | 0 | 1 | 203 |
| go through and avoid the rocks or other obstacles to reach the police car | 1 | 0 | 1 | 226 |
| go through and avoid the rocks or other obstacles to reach the red cube | 1 | 0 | 1 | 204 |
| go through and avoid the rocks or other obstacles to reach the red wooden bench | 1 | 0 | 1 | 173 |
| go through and avoid the rocks or other obstacles to reach the round wooden table | 1 | 0 | 1 | 242 |
| go through and avoid the stacked boxes or other obstacles to reach the black leather sofa | 1 | 0 | 1 | 166 |
| go through and avoid the stacked boxes or other obstacles to reach the blue barrel | 1 | 0 | 1 | 165 |
| go through and avoid the stacked boxes or other obstacles to reach the blue van | 1 | 0 | 1 | 174 |
| go through and avoid the stacked boxes or other obstacles to reach the green garbage truck | 1 | 0 | 1 | 203 |
| go through and avoid the stacked boxes or other obstacles to reach the man in blue jacket | 1 | 0 | 1 | 171 |
| go through and avoid the stacked boxes or other obstacles to reach the red barrel | 1 | 0 | 1 | 192 |
| go through and avoid the stacked boxes or other obstacles to reach the red sedan | 1 | 0 | 1 | 201 |
| go through and avoid the stacked boxes or other obstacles to reach the red wooden bench | 1 | 0 | 1 | 165 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow cylinder | 1 | 0 | 1 | 163 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow generator | 1 | 0 | 1 | 168 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow sphere | 1 | 0 | 1 | 176 |
| go through and avoid the stacked boxes or other obstacles to reach the yellow taxi | 1 | 0 | 1 | 174 |
| go through and avoid the standing stones or other obstacles to reach the ambulance | 1 | 0 | 1 | 241 |
| go through and avoid the standing stones or other obstacles to reach the beige sofa | 1 | 0 | 1 | 204 |
| go through and avoid the standing stones or other obstacles to reach the blue school chair | 1 | 0 | 1 | 172 |
| go through and avoid the standing stones or other obstacles to reach the blue van | 0 | 1 | 1 | 186 |
| go through and avoid the standing stones or other obstacles to reach the covered car | 1 | 0 | 1 | 170 |
| go through and avoid the standing stones or other obstacles to reach the girl in yellow shirt | 1 | 0 | 1 | 182 |
| go through and avoid the standing stones or other obstacles to reach the green cube | 1 | 0 | 1 | 166 |
| go through and avoid the standing stones or other obstacles to reach the green garbage truck | 1 | 0 | 1 | 228 |
| go through and avoid the standing stones or other obstacles to reach the green hatchback | 1 | 0 | 1 | 203 |
| go through and avoid the standing stones or other obstacles to reach the grey armchair | 1 | 0 | 1 | 171 |
| go through and avoid the standing stones or other obstacles to reach the man in blue jacket | 1 | 0 | 1 | 191 |
| go through and avoid the standing stones or other obstacles to reach the old television | 1 | 0 | 1 | 179 |
| go through and avoid the standing stones or other obstacles to reach the police car | 1 | 0 | 1 | 195 |
| go through and avoid the standing stones or other obstacles to reach the red barrel | 1 | 0 | 1 | 164 |
| go through and avoid the standing stones or other obstacles to reach the round wooden table | 1 | 0 | 1 | 179 |
| go through and avoid the standing stones or other obstacles to reach the white cabinet | 1 | 0 | 1 | 185 |
| go through and avoid the standing stones or other obstacles to reach the white pickup truck | 1 | 0 | 1 | 167 |
| go through and avoid the standing stones or other obstacles to reach the wooden barrel | 1 | 0 | 1 | 171 |
| go through and avoid the standing stones or other obstacles to reach the yellow taxi | 1 | 0 | 1 | 226 |
| go through and avoid the stone field or other obstacles to reach the black leather sofa | 1 | 0 | 1 | 217 |
| go through and avoid the stone field or other obstacles to reach the old television | 1 | 0 | 1 | 182 |
| go through and avoid the stone field or other obstacles to reach the orange cone | 1 | 0 | 1 | 176 |
| go through and avoid the stone field or other obstacles to reach the red barrel | 1 | 0 | 1 | 218 |
| go through and avoid the stone field or other obstacles to reach the red fire truck | 1 | 0 | 1 | 196 |
| go through and avoid the stone field or other obstacles to reach the white plastic chair | 1 | 0 | 1 | 188 |
| go through and avoid the stone field or other obstacles to reach the yellow cube | 1 | 0 | 1 | 182 |
| go through and avoid the stone field or other obstacles to reach the yellow cylinder | 1 | 0 | 1 | 162 |
| go through and avoid the stone field or other obstacles to reach the yellow suv | 1 | 0 | 1 | 173 |
| go through and avoid the stone field or other obstacles to reach the yellow wet floor sign | 1 | 0 | 1 | 203 |
| go through and avoid the tree clusters or other obstacles to reach the ambulance | 1 | 0 | 1 | 249 |
| go through and avoid the tree clusters or other obstacles to reach the black leather sofa | 1 | 0 | 1 | 186 |
| go through and avoid the tree clusters or other obstacles to reach the blue barrel | 1 | 0 | 1 | 209 |
| go through and avoid the tree clusters or other obstacles to reach the girl in yellow shirt | 1 | 0 | 1 | 159 |
| go through and avoid the tree clusters or other obstacles to reach the man in blue jacket | 1 | 0 | 1 | 160 |
| go through and avoid the tree clusters or other obstacles to reach the police car | 1 | 0 | 1 | 173 |
| go through and avoid the tree clusters or other obstacles to reach the red cylinder | 1 | 0 | 1 | 168 |
| go through and avoid the tree clusters or other obstacles to reach the red fire extinguisher | 1 | 0 | 1 | 173 |
| go through and avoid the tree clusters or other obstacles to reach the red tool chest | 1 | 0 | 1 | 178 |
| go through and avoid the tree clusters or other obstacles to reach the white cabinet | 1 | 0 | 1 | 177 |
| go through and avoid the tree clusters or other obstacles to reach the yellow cube | 1 | 0 | 1 | 184 |
| go through and avoid the tree clusters or other obstacles to reach the yellow wet floor sign | 1 | 0 | 1 | 169 |
| go through and avoid the trees or other obstacles to reach the blue cone | 1 | 0 | 1 | 160 |
| go through and avoid the trees or other obstacles to reach the blue school chair | 1 | 0 | 1 | 186 |
| go through and avoid the trees or other obstacles to reach the bronze whale statue | 1 | 0 | 1 | 182 |
| go through and avoid the trees or other obstacles to reach the green garbage truck | 1 | 0 | 1 | 204 |
| go through and avoid the trees or other obstacles to reach the man in red shirt | 1 | 0 | 1 | 169 |
| go through and avoid the trees or other obstacles to reach the person in purple shirt | 1 | 0 | 1 | 154 |
| go through and avoid the trees or other obstacles to reach the police officer | 1 | 0 | 1 | 199 |
| go through and avoid the trees or other obstacles to reach the red cylinder | 1 | 0 | 1 | 167 |
| go through and avoid the trees or other obstacles to reach the red sedan | 1 | 0 | 1 | 215 |
| go through and avoid the trees or other obstacles to reach the white cabinet | 1 | 0 | 1 | 168 |
| go through and avoid the trees or other obstacles to reach the white pickup truck | 1 | 0 | 1 | 178 |
| go through and avoid the trees or other obstacles to reach the yellow cylinder | 1 | 0 | 1 | 192 |
| go through and avoid the trees or other obstacles to reach the yellow generator | 1 | 0 | 1 | 168 |
| go through and avoid the trees or other obstacles to reach the yellow sphere | 1 | 0 | 1 | 231 |
| go through and avoid the trees or other obstacles to reach the yellow suv | 1 | 0 | 1 | 173 |
| go through and avoid the trees or other obstacles to reach the yellow taxi | 1 | 0 | 1 | 184 |
| go through and avoid the white pillars or other obstacles to reach the black leather sofa | 1 | 0 | 1 | 171 |
| go through and avoid the white pillars or other obstacles to reach the covered car | 1 | 0 | 1 | 157 |
| go through and avoid the white pillars or other obstacles to reach the man in brown jacket | 1 | 0 | 1 | 166 |

## Records

Each step holds `observation/image_0`, `observation/state` (float64 in TFDS, float32 values), `action` (float32) and `language_instruction`.

- image_0: RGB 256x256 PNG: the forward camera, one frame per 0.2 s step
- language_instruction: the episode's instruction, repeated on every step

Action:

| index | name | range | definition |
|---|---|---|---|
| 0 | forward_speed_m_s | [0.0, 2.0] | forward speed in the body frame, m/s |
| 1 | yaw_rate_rad_s | [-1.0, 1.0] | yaw rate, rad/s, positive turning right (clockwise seen from above) |
| 2 | vertical_speed_m_s | [-1.0, 1.0] | vertical speed, m/s, positive up |

State:

| index | name | status | definition |
|---|---|---|---|
| 0 | distance_to_target_m **PRIVILEGED** | spec_3_2 | horizontal distance from the drone to the target's centre, m |
| 1 | bearing_to_target_rad **PRIVILEGED** | not_autofly_verified | bearing of the target relative to the drone's heading, rad in [-pi, pi], positive to the right (\|r\| 0.92 / 0.88 with a motion-based estimate on the real episodes) |
| 2 | altitude_minus_1_47_m | adopted | altitude above ground minus 1.47 m (\|r\| 0.986 / 0.996; fitted 0.985 z - 1.470, 0.987 z - 1.477) |
| 3 | speed_m_s | spec_3_2 | speed: the norm of the velocity vector, m/s |
| 4 | vertical_velocity_m_s | not_autofly_verified | vertical velocity, up positive, m/s (\|r\| 0.77 / 0.60) |
| 5 | yaw_rate_rad_s | not_autofly_verified | yaw rate, rad/s, positive clockwise seen from above (\|r\| 0.54 / 0.76) |
| 6 | x_from_start_m | spec_3_2 | x (north) relative to the episode's start, m |
| 7 | y_from_start_m | spec_3_2 | y (east) relative to the episode's start, m |
| 8 | altitude_m | spec_3_2 | altitude above ground, m (z up) |

**PRIVILEGED**: state[0] (distance) and state[1] (bearing) are computed from the target's true position, which the drone cannot observe: a policy that reads them can fly to the target without the image or the instruction.

Episode metadata:

| field | type | meaning |
|---|---|---|
| file_path | text | <store>/episodes/<episode id>, as the release has it |
| episode_id | text | the episode's id in its store: <scene>_<seed> |
| store | text | the raw store (one scene and collection batch) it came from |
| scene | text | the scene id |
| split_of_scene | text | the scene's own split (train, test_seen, test_unseen); not the dataset split |
| target_name | text | the target the instruction names |
| target_category | text | vehicle, geometry, furniture, equipment, human or other (the paper's Fig. 3c mix) |
| target_seen | bool | whether the target is in the seen pool (unseen targets are for test_unseen scenes) |
| target_status | text | the pool entry's status: primitive (engine shape), imported (a real model) or placeholder |
| target_stand_in | bool | a low-poly stand-in (Kenney) for the object its name describes; filter on it to drop them |
| seed | int64 | the episode seed (start, target, distractors and layout follow from it and the scene) |
| pilot | text | sha256 of the expert checkpoint that flew it |

## Split

val_fraction 0.05, seed 0, stratified by scene x target_name; frozen in `data/autofly_ue5_proto_split.json` (sha256 `623bd6d11b85bb2609b2c30af41a6103b41329a1fb7e74a07028eddab717c70e`).

val takes round(val_fraction x episodes), at least one, given out one at a time to the scene x target stratum furthest below its share (val_fraction x its episodes; ties by sha256 of seed and stratum), never a stratum's last train episode; within a stratum its new episodes go in sha256(seed, key) order. Episodes already in the file keep their split.

## Normalisation statistics

`dataset_statistics.json` in OpenVLA's schema, over the train split only (1710 episodes, 322125 records):

|  | action[3] | proprio = state[9] |
|---|---|---|
| mean | [1.876, 0.0002098, -0.01392] | [38.19, -0.01529, 0.5005, 1.877, 0.0005795, 0.002494, 0.4769, 0.9757, 1.971] |
| std | [0.3431, 0.663, 0.5393] | [19.96, 0.3409, 0.2223, 0.326, 0.4974, 0.3004, 27.03, 27.53, 0.2223] |
| min | [1.591e-05, -1, -0.9999] | [1.095, -3.14, -0.4684, 1.231e-06, -1.348, -0.9655, -66.36, -66.63, 1.002] |
| max | [2, 1, 0.9999] | [93.81, 3.137, 1.528, 2.341, 1.482, 0.9888, 67.49, 66.47, 2.998] |
| q01 | [0.02718, -0.9902, -0.9512] | [5.245, -1.02, -0.03035, 0.3311, -0.9755, -0.6935, -58.67, -59.23, 1.44] |
| q99 | [1.998, 0.9911, 0.9575] | [79.28, 0.8598, 1.062, 2.183, 1.042, 0.7154, 58.89, 58.94, 2.532] |

## Provenance

| store | episodes | records | rejects | kept-rate | scenes | created | pilot sha256 |
|---|---|---|---|---|---|---|---|
| s01_c1 | 50 | 9747 | 7 | 0.8772 | s01 | 2026-10-09 01:21:35 | 8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc |
| s01_c2 | 50 | 9699 | 2 | 0.9615 | s01 | 2026-10-09 01:23:05 | 8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc |
| s01_c3 | 50 | 9715 | 1 | 0.9804 | s01 | 2026-10-09 01:24:35 | 8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc |
| s01_c4 | 50 | 9707 | 2 | 0.9615 | s01 | 2026-10-09 01:25:05 | 8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc |
| s01d_c1 | 50 | 9845 | 6 | 0.8929 | s01d | 2026-10-09 00:33:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s01d_c2 | 50 | 9680 | 5 | 0.9091 | s01d | 2026-10-09 01:11:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s01d_c3 | 50 | 10171 | 11 | 0.8197 | s01d | 2026-10-09 01:17:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s01d_c4 | 50 | 9876 | 6 | 0.8929 | s01d | 2026-10-09 01:19:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s02_c1 | 50 | 9258 | 12 | 0.8065 | s02 | 2026-10-09 00:25:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s02_c2 | 50 | 9123 | 12 | 0.8065 | s02 | 2026-10-09 00:27:05 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s02_c3 | 50 | 8890 | 16 | 0.7576 | s02 | 2026-10-09 00:27:35 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s02_c4 | 50 | 9341 | 14 | 0.7812 | s02 | 2026-10-09 00:31:05 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s03_c1 | 50 | 10278 | 28 | 0.641 | s03 | 2026-10-08 23:26:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s03_c2 | 50 | 9480 | 28 | 0.641 | s03 | 2026-10-08 23:35:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s03_c3 | 50 | 9713 | 27 | 0.6494 | s03 | 2026-10-09 00:10:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s03_c4 | 50 | 9666 | 29 | 0.6329 | s03 | 2026-10-09 00:24:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s04_c1 | 50 | 9737 | 45 | 0.5263 | s04 | 2026-10-08 23:12:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s04_c2 | 50 | 9494 | 41 | 0.5495 | s04 | 2026-10-08 23:18:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s04_c3 | 50 | 9455 | 46 | 0.5208 | s04 | 2026-10-08 23:21:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s04_c4 | 50 | 9127 | 42 | 0.5435 | s04 | 2026-10-08 23:22:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s05_c1 | 50 | 9681 | 53 | 0.4854 | s05 | 2026-10-08 22:04:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s05_c2 | 50 | 9667 | 37 | 0.5747 | s05 | 2026-10-08 22:05:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s05_c3 | 50 | 9489 | 58 | 0.463 | s05 | 2026-10-08 22:05:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s05_c4 | 50 | 9299 | 34 | 0.5952 | s05 | 2026-10-08 23:10:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s06_c1 | 50 | 9306 | 79 | 0.3876 | s06 | 2026-10-08 22:02:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s06_c2 | 50 | 9489 | 94 | 0.3472 | s06 | 2026-10-08 22:03:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s06_c3 | 50 | 9192 | 84 | 0.3731 | s06 | 2026-10-08 22:03:34 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s06_c4 | 50 | 9079 | 76 | 0.3968 | s06 | 2026-10-08 22:04:04 | 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 |
| s07_c1 | 50 | 8751 | 0 | 1.0 | s07 | 2026-10-09 01:55:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s07_c2 | 50 | 8836 | 4 | 0.9259 | s07 | 2026-10-09 02:05:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s07_c3 | 50 | 8928 | 0 | 1.0 | s07 | 2026-10-09 02:06:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s07_c4 | 50 | 8734 | 0 | 1.0 | s07 | 2026-10-09 02:08:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s09_c1 | 50 | 9031 | 7 | 0.8772 | s09 | 2026-10-09 02:08:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s09_c2 | 50 | 9163 | 1 | 0.9804 | s09 | 2026-10-09 02:10:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s09_c3 | 50 | 9218 | 2 | 0.9615 | s09 | 2026-10-09 02:10:35 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |
| s09_c4 | 50 | 9142 | 0 | 1.0 | s09 | 2026-10-09 02:33:05 | 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd |

Pilots:

| sha256 | episodes | stores | checkpoint | deterministic |
|---|---|---|---|---|
| 8c2c7500041907a98d5d696b9007e30578731ff5136f63c384270c0f4fcbf8dc | 200 | s01_c1, s01_c2, s01_c3, s01_c4 | runs/expert/s01_r2/best/best_model.zip | False |
| 55e666a555800de5a1fe97c89de58a667b333d45c2cb17a4ff6ce1c3944efbfd | 600 | s01d_c1, s01d_c2, s01d_c3, s01d_c4, s07_c1, s07_c2, s07_c3, s07_c4, s09_c1, s09_c2, s09_c3, s09_c4 | runs/expert/s01d_r6/averaged/avg_250k_550k.zip | False |
| 0971010abb0512c2660c4dd33ad7f75e8d19acbcbb400d38935ba12c08f3ca73 | 1000 | s02_c1, s02_c2, s02_c3, s02_c4, s03_c1, s03_c2, s03_c3, s03_c4, s04_c1, s04_c2, s04_c3, s04_c4, s05_c1, s05_c2, s05_c3, s05_c4, s06_c1, s06_c2, s06_c3, s06_c4 | runs/expert/s02_r1/averaged/avg_350k_end.zip | False |

Scene configs:

| config | sha256 |
|---|---|
| scene_autofly_s01_fast.jsonc | f3fe5ac8a23faefc58186b3fc8d6aa3134d01346431bd21c50b63d0b24b77c94 |
| scene_autofly_s02_fast.jsonc | 2ba549af66881da610be7c47c58fa936ac7ca022ec4a52d9d7541ec98ca35ebc |
| scene_autofly_s03_fast.jsonc | 1775b4bcd8d106ad4c92e5c74f1ec5ef342fc9b8cf68c8e899f9bad116e256fb |
| scene_autofly_s04_fast.jsonc | b0a834fa548ba99d5170df142b8d8a9ba89f74e17305f5d33baa6511c71f0ba9 |
| scene_autofly_s05_fast.jsonc | 1c3fbd1e22526e5e72466f7540f3cfa015e6ac2ed1decc1c587e5ce550d7fa87 |
| scene_autofly_s06_fast.jsonc | 237ca7b4ae58f38f9513a51aca510389490528fda11c3eeb561c8790c1b2603a |
| scene_autofly_s07_fast.jsonc | ff609452edf4de6a505405318df18a743579bdf59a43a2186c665d7b552fb51e |
| scene_autofly_s09_fast.jsonc | c5e71b6be7ec93225278b78e2f6cf78714ba12d9596daa9007a606acc27bd92e |

Collector git heads:

| head | episodes |
|---|---|
| 0388392a0c78f86a5c61b90a4d62327e1ed79917 | 1450 |
| 1f66cc4965af46aab9fa640ea27511f7f5002bda | 350 |

Exported at git head e88ed92f6042eb442c0522347ef96aa352c5e084 (tree dirty: True).

## Known limits

- Language: the two instruction templates are AutoFly's, verbatim, including the original's spelling "avioding" ('advance to {target} while avioding {obstacle} and other obstacles'); with one obstacle phrase per scene, every instruction is a function of the scene and the target. No paraphrases.
- The state is partly privileged: state[0] (distance) and state[1] (bearing) are computed from the target's true position, which the drone cannot observe: a policy that reads them can fly to the target without the image or the instruction. They are AutoFly's own definition, kept as the record's; zero or drop them for a policy that should navigate from the image and the instruction.
- state[1] bearing_to_target_rad, state[4] vertical_velocity_m_s, state[5] yaw_rate_rad_s follow best-fit definitions that could not be verified against AutoFly's released episodes (status not_autofly_verified; dataset/state.py).
- Kenney low-poly stand-ins stand for vehicles and people where no photoreal model was available (target_stand_in; 682 episodes here). A later Fab session may replace them.
- Only successful, collision-free episodes are kept; failures stay in each store's rejects. The SAC experts are below the paper's 0.95 success on several scenes, so each scene's data covers the starts its expert could fly (kept-rates: s01_c1 0.8772, s01_c2 0.9615, s01_c3 0.9804, s01_c4 0.9615, s01d_c1 0.8929, s01d_c2 0.9091, s01d_c3 0.8197, s01d_c4 0.8929, s02_c1 0.8065, s02_c2 0.8065, s02_c3 0.7576, s02_c4 0.7812, s03_c1 0.641, s03_c2 0.641, s03_c3 0.6494, s03_c4 0.6329, s04_c1 0.5263, s04_c2 0.5495, s04_c3 0.5208, s04_c4 0.5435, s05_c1 0.4854, s05_c2 0.5747, s05_c3 0.463, s05_c4 0.5952, s06_c1 0.3876, s06_c2 0.3472, s06_c3 0.3731, s06_c4 0.3968, s07_c1 1.0, s07_c2 0.9259, s07_c3 1.0, s07_c4 1.0, s09_c1 0.8772, s09_c2 0.9804, s09_c3 0.9615, s09_c4 1.0). The actions are the expert's (stochastic where the pilot record says deterministic false), not a human's.
- The val split is a stratified random sample of the same scenes and targets: an in-distribution validation set, not a test of unseen scenes or targets (test_seen and test_unseen are evaluated in closed loop).
- Per-step phase weights from the rebalancing stage (spec §10.3) are not part of the records.

## Loading

- TFDS: tfds.builder_from_directory('<out>/autofly_ue5_proto/1.0.0'), or tfds.builder('autofly_ue5_proto', data_dir='<out>') as OpenVLA calls it
- OpenVLA-style: image_obs_keys {'primary': 'image_0'}, state_obs_keys ['state'], language_key 'language_instruction'; the action is 3-dim, so pass action_normalization_mask [True]*3 and no absolute_action_mask (OpenVLA's make_oxe_dataset_kwargs assumes 7-dim EEF actions); dataset_statistics.json[name] is the normalisation (BOUNDS_Q99 uses q01 and q99)
