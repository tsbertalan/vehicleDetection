# Requires apt package markdown
report.html:
	cat style.html > report.html
	echo "<article class=\"markdown-body\">" >> report.html
	markdown report.md >> report.html
	echo "</article class=\"markdown-body\">" >> report.html
	echo "</body></html>" >> report.html
view:
	google-chrome report.html &

.PHONY: report.pdf report.html
