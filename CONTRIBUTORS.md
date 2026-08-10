# Welcome to PySR's contributing guide <!-- Forked from GitHub's awesome template contributing guide -->

Thank you for investing your time in contributing to our project! Any contribution you make will be reflected on the [contributors list](#contributors) :sparkles:.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

## New contributor guide

To get an overview of the project, read PySR's [README](README.md). The [PySR docs](https://ai.damtp.cam.ac.uk/pysr/) give additional information.
Here are some resources to help you get started with open source contributions in general:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

### Issues

#### Create a new issue

If you spot a problem with PySR, [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments). If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/astroautomata/PySR/issues/new/choose).

#### Solve an issue

Scan through our [existing issues](https://github.com/astroautomata/PySR/issues) to find one that interests you (feel free to work on any!). You can narrow down the search using `labels` as filters. See [Labels](/contributing/how-to-use-labels.md) for more information. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make Changes

#### Make changes locally

1. Fork the repository.
- Using GitHub Desktop:
  - [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!

- Using the command line:
  - [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.

2. Create a working branch and start with your changes!

3. (Optional) If you would like to make changes to PySR itself, skip to step 4. However, if you are interested in making changes to the _symbolic regression code_ itself,
check out the [guide](https://ai.damtp.cam.ac.uk/pysr/backend/) on modifying a custom SymbolicRegression.jl library.
In this case, you might instead be interested in making suggestions to the [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl) library.

4. You can install your local version of PySR with `pip install -e '.[dev]'`, and run tests with `python -m pysr test main`.

### Commit your update

Once you are happy with your changes, run `black .` to apply [Black](https://black.readthedocs.io/en/stable/) formatting to your local
version. Commit the changes once you are ready.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a PySR team member will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations :tada::tada: The PySR team thanks you :sparkles:.

Once your PR is merged, your contributions will be publicly visible.

Thanks for being part of the PySR community!

<div align="center">

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adil-soubki"><img src="https://avatars.githubusercontent.com/u/5231841?v=4?s=50" width="50px;" alt="Adil"/><br /><sub><b>Adil</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://cjdoris.github.io/"><img src="https://avatars.githubusercontent.com/u/1844215?v=4?s=50" width="50px;" alt="Christopher Rowley"/><br /><sub><b>Christopher Rowley</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/markkittisopikul/"><img src="https://avatars.githubusercontent.com/u/8062771?v=4?s=50" width="50px;" alt="Mark Kittisopikul"/><br /><sub><b>Mark Kittisopikul</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tttc3"><img src="https://avatars.githubusercontent.com/u/97948946?v=4?s=50" width="50px;" alt="T Coxon"/><br /><sub><b>T Coxon</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DhananjayAshok"><img src="https://avatars.githubusercontent.com/u/46792537?v=4?s=50" width="50px;" alt="Dhananjay Ashok"/><br /><sub><b>Dhananjay Ashok</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/johanbluecreek"><img src="https://avatars.githubusercontent.com/u/852554?v=4?s=50" width="50px;" alt="Johan Blåbäck"/><br /><sub><b>Johan Blåbäck</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://mathopt.de/people/martensen/index.php"><img src="https://avatars.githubusercontent.com/u/20998300?v=4?s=50" width="50px;" alt="JuliusMartensen"/><br /><sub><b>JuliusMartensen</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ngam"><img src="https://avatars.githubusercontent.com/u/67342040?v=4?s=50" width="50px;" alt="ngam"/><br /><sub><b>ngam</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kazewong"><img src="https://avatars.githubusercontent.com/u/8803931?v=4?s=50" width="50px;" alt="Kaze Wong"/><br /><sub><b>Kaze Wong</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ChrisRackauckas"><img src="https://avatars.githubusercontent.com/u/1814174?v=4?s=50" width="50px;" alt="Christopher Rackauckas"/><br /><sub><b>Christopher Rackauckas</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://kidger.site/"><img src="https://avatars.githubusercontent.com/u/33688385?v=4?s=50" width="50px;" alt="Patrick Kidger"/><br /><sub><b>Patrick Kidger</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/OkonSamuel"><img src="https://avatars.githubusercontent.com/u/39421418?v=4?s=50" width="50px;" alt="Okon Samuel"/><br /><sub><b>Okon Samuel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/w2ll2am"><img src="https://avatars.githubusercontent.com/u/16038228?v=4?s=50" width="50px;" alt="William Booth-Clibborn"/><br /><sub><b>William Booth-Clibborn</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ayagh19"><img src="https://avatars.githubusercontent.com/u/124587945?v=4?s=50" width="50px;" alt="Aya Ghaleb"/><br /><sub><b>Aya Ghaleb</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/gca30"><img src="https://avatars.githubusercontent.com/u/124273598?v=4?s=50" width="50px;" alt="gca30"/><br /><sub><b>gca30</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/nmheim"><img src="https://avatars.githubusercontent.com/u/29552345?v=4?s=50" width="50px;" alt="Niklas Heim"/><br /><sub><b>Niklas Heim</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/atharvas"><img src="https://avatars.githubusercontent.com/u/20322919?v=4?s=50" width="50px;" alt="Atharva Sehgal"/><br /><sub><b>Atharva Sehgal</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/wkharold"><img src="https://avatars.githubusercontent.com/u/103685?v=4?s=50" width="50px;" alt="wkharold"/><br /><sub><b>wkharold</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://wsmoses.com"><img src="https://avatars.githubusercontent.com/u/1260124?v=4?s=50" width="50px;" alt="William Moses"/><br /><sub><b>William Moses</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://grezde.github.io"><img src="https://avatars.githubusercontent.com/u/43924925?v=4?s=50" width="50px;" alt="Ardeleanu Cristian"/><br /><sub><b>Ardeleanu Cristian</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/gm89uk"><img src="https://avatars.githubusercontent.com/u/127948719?v=4?s=50" width="50px;" alt="gm89uk"/><br /><sub><b>gm89uk</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://pablo-lemos.github.io/"><img src="https://avatars.githubusercontent.com/u/38078898?v=4?s=50" width="50px;" alt="Pablo Lemos"/><br /><sub><b>Pablo Lemos</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Moelf"><img src="https://avatars.githubusercontent.com/u/5306213?v=4?s=50" width="50px;" alt="Jerry Ling"/><br /><sub><b>Jerry Ling</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/CharFox1"><img src="https://avatars.githubusercontent.com/u/35052672?v=4?s=50" width="50px;" alt="Charles Fox"/><br /><sub><b>Charles Fox</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/johannbrehmer"><img src="https://avatars.githubusercontent.com/u/17068560?v=4?s=50" width="50px;" alt="Johann Brehmer"/><br /><sub><b>Johann Brehmer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.cosmicmar.com/"><img src="https://avatars.githubusercontent.com/u/1510968?v=4?s=50" width="50px;" alt="Marius Millea"/><br /><sub><b>Marius Millea</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://gitlab.com/cobac"><img src="https://avatars.githubusercontent.com/u/27872944?v=4?s=50" width="50px;" alt="Coba"/><br /><sub><b>Coba</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/foxtran"><img src="https://avatars.githubusercontent.com/u/39676482?v=4?s=50" width="50px;" alt="foxtran"/><br /><sub><b>foxtran</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://smhasan.com/"><img src="https://avatars.githubusercontent.com/u/36223598?v=4?s=50" width="50px;" alt="Shah Mahdi Hasan "/><br /><sub><b>Shah Mahdi Hasan </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://aluthge.com"><img src="https://avatars.githubusercontent.com/u/5619885?v=4?s=50" width="50px;" alt="Dilum Aluthge"/><br /><sub><b>Dilum Aluthge</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/SebastianM-C"><img src="https://avatars.githubusercontent.com/u/31181429?v=4?s=50" width="50px;" alt="Sebastian Micluța-Câmpeanu"/><br /><sub><b>Sebastian Micluța-Câmpeanu</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://neuroscience.wustl.edu/people/timothy-holy-phd/"><img src="https://avatars.githubusercontent.com/u/1525481?v=4?s=50" width="50px;" alt="Tim Holy"/><br /><sub><b>Tim Holy</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/BrotherHa"><img src="https://avatars.githubusercontent.com/u/190199534?v=4?s=50" width="50px;" alt="BrotherHa"/><br /><sub><b>BrotherHa</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://wthompson.space"><img src="https://avatars.githubusercontent.com/u/7330605?v=4?s=50" width="50px;" alt="William Thompson"/><br /><sub><b>William Thompson</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://abzu.ai"><img src="https://avatars.githubusercontent.com/u/2547785?v=4?s=50" width="50px;" alt="Tom Jelen"/><br /><sub><b>Tom Jelen</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.miguelromao.me/"><img src="https://avatars.githubusercontent.com/u/7794475?v=4?s=50" width="50px;" alt="Miguel Crispim Romao"/><br /><sub><b>Miguel Crispim Romao</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adienes"><img src="https://avatars.githubusercontent.com/u/51664769?v=4?s=50" width="50px;" alt="Andy Dienes"/><br /><sub><b>Andy Dienes</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://singhharsh.in"><img src="https://avatars.githubusercontent.com/u/143034341?v=4?s=50" width="50px;" alt="Harsh Singh "/><br /><sub><b>Harsh Singh </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pitmonticone"><img src="https://avatars.githubusercontent.com/u/38562595?v=4?s=50" width="50px;" alt="Pietro Monticone"/><br /><sub><b>Pietro Monticone</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sheevy"><img src="https://avatars.githubusercontent.com/u/1525683?v=4?s=50" width="50px;" alt="Mateusz Kubica"/><br /><sub><b>Mateusz Kubica</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/WilliamBC-SL"><img src="https://avatars.githubusercontent.com/u/118170949?v=4?s=50" width="50px;" alt="William Booth-Clibborn"/><br /><sub><b>William Booth-Clibborn</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://raulpl.github.io/about"><img src="https://avatars.githubusercontent.com/u/3116652?v=4?s=50" width="50px;" alt="Raúl Peralta Lozada"/><br /><sub><b>Raúl Peralta Lozada</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/hvaara/"><img src="https://avatars.githubusercontent.com/u/1535968?v=4?s=50" width="50px;" alt="Roy Hvaara"/><br /><sub><b>Roy Hvaara</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/VishalJ99"><img src="https://avatars.githubusercontent.com/u/51826812?v=4?s=50" width="50px;" alt="Vishal Jain"/><br /><sub><b>Vishal Jain</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/spaette"><img src="https://avatars.githubusercontent.com/u/111918424?v=4?s=50" width="50px;" alt="spaette"/><br /><sub><b>spaette</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.yxliu.group"><img src="https://avatars.githubusercontent.com/u/1089344?v=4?s=50" width="50px;" alt="Yi-Xin Liu"/><br /><sub><b>Yi-Xin Liu</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/spinnau"><img src="https://avatars.githubusercontent.com/u/2995937?v=4?s=50" width="50px;" alt="spinnau"/><br /><sub><b>spinnau</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sunxd3"><img src="https://avatars.githubusercontent.com/u/5433119?v=4?s=50" width="50px;" alt="Xianda Sun"/><br /><sub><b>Xianda Sun</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://jaywadekar.github.io/"><img src="https://avatars.githubusercontent.com/u/5493388?v=4?s=50" width="50px;" alt="Jay Wadekar"/><br /><sub><b>Jay Wadekar</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ablaom"><img src="https://avatars.githubusercontent.com/u/30517088?v=4?s=50" width="50px;" alt="Anthony Blaom, PhD"/><br /><sub><b>Anthony Blaom, PhD</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jgmedina95"><img src="https://avatars.githubusercontent.com/u/97254349?v=4?s=50" width="50px;" alt="Jgmedina95"/><br /><sub><b>Jgmedina95</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/mcabbott"><img src="https://avatars.githubusercontent.com/u/32575566?v=4?s=50" width="50px;" alt="Michael Abbott"/><br /><sub><b>Michael Abbott</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/oscardssmith"><img src="https://avatars.githubusercontent.com/u/11729272?v=4?s=50" width="50px;" alt="Oscar Smith"/><br /><sub><b>Oscar Smith</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://ericphanson.com/"><img src="https://avatars.githubusercontent.com/u/5846501?v=4?s=50" width="50px;" alt="Eric Hanson"/><br /><sub><b>Eric Hanson</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/henriquebecker91"><img src="https://avatars.githubusercontent.com/u/14113435?v=4?s=50" width="50px;" alt="Henrique Becker"/><br /><sub><b>Henrique Becker</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/qwertyjl"><img src="https://avatars.githubusercontent.com/u/110912592?v=4?s=50" width="50px;" alt="qwertyjl"/><br /><sub><b>qwertyjl</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://huijzer.xyz/"><img src="https://avatars.githubusercontent.com/u/20724914?v=4?s=50" width="50px;" alt="Rik Huijzer"/><br /><sub><b>Rik Huijzer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/GCaptainNemo"><img src="https://avatars.githubusercontent.com/u/43086239?v=4?s=50" width="50px;" alt="Hongyu Wang"/><br /><sub><b>Hongyu Wang</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/ZehaoJin"><img src="https://avatars.githubusercontent.com/u/50961376?v=4?s=50" width="50px;" alt="Zehao Jin"/><br /><sub><b>Zehao Jin</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/tmengel"><img src="https://avatars.githubusercontent.com/u/38924390?v=4?s=50" width="50px;" alt="Tanner Mengel"/><br /><sub><b>Tanner Mengel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/agrundner24"><img src="https://avatars.githubusercontent.com/u/38557656?v=4?s=50" width="50px;" alt="Arthur Grundner"/><br /><sub><b>Arthur Grundner</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sjwetzel"><img src="https://avatars.githubusercontent.com/u/24393721?v=4?s=50" width="50px;" alt="sjwetzel"/><br /><sub><b>sjwetzel</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://sauravmaheshkar.github.io/"><img src="https://avatars.githubusercontent.com/u/61241031?v=4?s=50" width="50px;" alt="Saurav Maheshkar"/><br /><sub><b>Saurav Maheshkar</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/chris-soelistyo"><img src="https://avatars.githubusercontent.com/u/68875981?v=4?s=50" width="50px;" alt="chris-soelistyo"/><br /><sub><b>chris-soelistyo</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://ilyaorson.gitlab.io"><img src="https://avatars.githubusercontent.com/u/12092488?v=4?s=50" width="50px;" alt="Ilya Orson "/><br /><sub><b>Ilya Orson </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://hftsoi.github.io"><img src="https://avatars.githubusercontent.com/u/51976330?v=4?s=50" width="50px;" alt="Ho Fung Tsoi"/><br /><sub><b>Ho Fung Tsoi</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/LionessOfCintra"><img src="https://avatars.githubusercontent.com/u/92221853?v=4?s=50" width="50px;" alt="LionessOfCintra"/><br /><sub><b>LionessOfCintra</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/manuel-morales-a"><img src="https://avatars.githubusercontent.com/u/64017590?v=4?s=50" width="50px;" alt="Manuel Morales "/><br /><sub><b>Manuel Morales </b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://paulomontero.github.io"><img src="https://avatars.githubusercontent.com/u/23636178?v=4?s=50" width="50px;" alt="Paulo Montero Camacho"/><br /><sub><b>Paulo Montero Camacho</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/luna026"><img src="https://avatars.githubusercontent.com/u/88938665?v=4?s=50" width="50px;" alt="Writu Dasgupta"/><br /><sub><b>Writu Dasgupta</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/anubhavkamal"><img src="https://avatars.githubusercontent.com/u/23038512?v=4?s=50" width="50px;" alt="Anubhav Kamal"/><br /><sub><b>Anubhav Kamal</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/anthony-sun"><img src="https://avatars.githubusercontent.com/u/115842064?v=4?s=50" width="50px;" alt="anthony-sun"/><br /><sub><b>anthony-sun</b></sub></a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://nithouson.github.io"><img src="https://avatars.githubusercontent.com/u/26868834?v=4?s=50" width="50px;" alt="Hao Guo"/><br /><sub><b>Hao Guo</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/TrailblazerH"><img src="https://avatars.githubusercontent.com/u/177746076?v=4?s=50" width="50px;" alt="Trailblazer"/><br /><sub><b>Trailblazer</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/christospliakos"><img src="https://avatars.githubusercontent.com/u/64842094?v=4?s=50" width="50px;" alt="Christos Pliakos"/><br /><sub><b>Christos Pliakos</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/zouzaxd"><img src="https://avatars.githubusercontent.com/u/103605983?v=4?s=50" width="50px;" alt="Sousa Neto"/><br /><sub><b>Sousa Neto</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/LeoVoltolini"><img src="https://avatars.githubusercontent.com/u/94749527?v=4?s=50" width="50px;" alt="Leonardo Voltolini"/><br /><sub><b>Leonardo Voltolini</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/con13375"><img src="https://avatars.githubusercontent.com/u/19805622?v=4?s=50" width="50px;" alt="Daniel Eduardo Conde Villatoro"/><br /><sub><b>Daniel Eduardo Conde Villatoro</b></sub></a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sambeckers"><img src="https://avatars.githubusercontent.com/u/127021792?v=4?s=50" width="50px;" alt="Sam Beckers"/><br /><sub><b>Sam Beckers</b></sub></a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
</div>
