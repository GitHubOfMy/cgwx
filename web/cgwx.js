import { app } from "/scripts/app.js";

app.registerExtension({
    name: "CGWX",
    settings: [
        {
            id: "CGWX.ProjectDescription1",
            category: ["CGWX","Project1"], 
            name: "Project Description1", 
            tooltip: "Project Description1",
			//后面必须跟一个类型，不写默认一般是输入框
			type: "boolean",
			defaultValue: true
        },
		{
            id: "CGWX.ProjectDescription2",
            category: ["CGWX","Project2"], 
            name: "Project Description2", 
            tooltip: "Project Description2",
			type: "boolean",
			defaultValue: true,
			//icon烧杯显示，提醒用户这个功能是试验性的
			experimental: true
        }
    ],

    // 设置-关于  最上面一行显示的
    aboutPageBadges: [
        {
            label: "CGWX",
            url: "https://www.wsxhome.cn",
            icon: "pi pi-globe"
        }
    ],
});