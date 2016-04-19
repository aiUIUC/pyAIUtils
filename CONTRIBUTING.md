# Contributing to aiutils

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

This is a list of suggestions for contributing to aiutils. These are just suggestions, not rules, use your best judgment and feel free to propose changes to this document in a pull request.

## A basic set of commands for using git collaboratively
This is basically a summary of the articles from [bocoup](https://bocoup.com/weblog/git-workflow-walkthrough-feature-branches)

### Start a new branch from master
```
git checkout master
git pull origin master
git checkout -b BRANCH
```

### Push the branch to remote
```
git push origin BRANCH
```

### To pull down a branch
```
git checkout -b BRANCH origin/BRANCH
```

### Easy way to fix merge conflicts
```
git fetch origin master
git checkout BRANCH
git merge master
```

Then fix conflicts, and push the branch again.

```
git push origin BRANCH
```

#### Delete remote branch
```
git push orign :BRANCH
```

## Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally
* Consider starting the commit message with an applicable emoji:
    * :art: `:art:` when improving the format/structure of the code
    * :racehorse: `:racehorse:` when improving performance
    * :memo: `:memo:` when writing docs
    * :bug: `:bug:` when fixing a bug
    * :fire: `:fire:` when removing code or files
    * :white_check_mark: `:white_check_mark:` when adding tests
    * :shirt: `:shirt:` when removing linter warnings